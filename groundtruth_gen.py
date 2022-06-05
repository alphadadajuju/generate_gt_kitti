from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import pickle
import numpy as np
import sys
import cv2
from matplotlib import pyplot as plt

from ACT_utils import nms_tubelets, iou2d


def BuildGT():
    
    """
    Abstract class for handling dataset of tubes.
    
    Here we assume that a pkl file exists as a cache. The cache is a dictionary with the following keys:
        labels: list of labels
        train_videos: a list with nsplits elements, each one containing the list of training videos
        test_videos: idem for the test videos
        nframes: dictionary that gives the number of frames for each video
        resolution: dictionary that output a tuple (h,w) of the resolution for each video
        gttubes: dictionary that contains the gt tubes for each video.
                    Gttubes are dictionary that associates from each index of label, a list of tubes.
                    A tube is a numpy array with nframes rows and 5 columns, <frame number> <x1> <y1> <x2> <y2>.
    """
    
    
    # some configs for BuildGT
    K = 1 # link one frame at a time
    root_path = '/home/judy/Bureau/Dataset/'
    train_test = os.listdir(root_path) # train_test[0]: 'Training'; train_test[1]: 'Test'
    classes = ['twilight', 'day', 'night']
    
    GT = {} # final pkl file to store
    GT['labels'] = ['Moving']
    
    # gather train/test video directories
    vlist = []
    v_train, v_test = [], []
    
    for tt in train_test:
        
        for cc in classes:
            
            v_path = os.path.join(root_path, tt, cc) 
            
            if not os.path.isdir(v_path):
                continue
            
            v_folder = os.listdir(v_path)
            vlist.extend([v_path + '/' + v_folder[i] for i in range(len(v_folder))])
            
            if tt == 'Training':
                v_train.extend(v_folder)
            else:
                v_test.extend(v_folder)
    
    GT['Train_videos'] = [v_train]
    GT['Test_videos'] = [v_test]
    
    # link frame-wise detection into tubes;
    # codes modified from ACT_build.py 
    gttubes = {}
    nframes_dict ={}
    resolution = {}
    for iv, v in enumerate(vlist):
        
        v_boxpath = v + '/Boxes'
        RES = {}
        all_files = os.listdir(v_boxpath)
        all_files.sort()
    
        nframes = len(all_files)
        
        # load detected tubelets
        VDets = {}
        for f, f_name in enumerate(all_files):
            
            f_name_full = v_boxpath + '/' + f_name
            with open(f_name_full, 'rt') as fd:
                lines = fd.readlines()
                dets = []
                for l in lines:
                  coords = l.split(' ')  
                  # x1, y1, x2, y2, score 
                  det = [float(coords[0]), float(coords[1]), float(coords[2]), float(coords[3]), 1.0]
                  dets.append(det)
                
            VDets[f+1] = {1: np.array(dets, np.float32)} # 1: fake label class
                
        for ilabel in range(1):
            FINISHED_TUBES = []
            CURRENT_TUBES = []  # tubes is a list of tuple (frame, lstubelets)
            # calculate average scores of tubelets in tubes

            def tubescore(tt):
                return np.mean(np.array([tt[i][1][-1] for i in range(len(tt))]))

            for frame in range(1, nframes+1):
                # load boxes of the new frame and do nms while keeping Nkeep highest scored
                ltubelets = VDets[frame][ilabel + 1]  # [:,range(4*K) + [4*K + 1 + ilabel]]  Nx(4K+1) with (x1 y1 x2 y2)*K ilabel-score

                ltubelets = nms_tubelets(ltubelets, 0.6, top_k=10)
                # TODO: start with the second frame according to Judy's spec
                # just start new tubes 
                if frame == 1:
                    for i in range(ltubelets.shape[0]):
                        CURRENT_TUBES.append([(1, ltubelets[i, :])])
                    continue

                # sort current tubes according to average score
                avgscore = [tubescore(t) for t in CURRENT_TUBES]
                argsort = np.argsort(-np.array(avgscore))
                CURRENT_TUBES = [CURRENT_TUBES[i] for i in argsort]
                # loop over tubes
                finished = []
                for it, t in enumerate(CURRENT_TUBES):
                    # compute ious between the last box of t and ltubelets
                    last_frame, last_tubelet = t[-1]
                    ious = []
                    offset = frame - last_frame
                    if offset < K:
                        nov = K - offset
                        ious = sum([iou2d(ltubelets[:, 4 * iov:4 * iov + 4], last_tubelet[4 * (iov + offset):4 * (iov + offset + 1)]) for iov in range(nov)]) / float(nov)
                    else:
                        ious = iou2d(ltubelets[:, :4], last_tubelet[4 * K - 4:4 * K])

                    #valid = np.where(ious >= 0.5)[0]
                    valid = None
                    try: 
                        if len(ious) != 0 and max(ious) >= 0.5:
                            valid = np.array([np.argmax(ious)])
                    except ValueError:
                        print('')
                        
                    if valid is not None: #valid.size > 0:
                        # take the one with maximum score
                        #idx = valid[np.argmax(ltubelets[valid, -1])]
                        idx = valid[0]
                        try:
                            CURRENT_TUBES[it].append((frame, ltubelets[idx, :]))
                        
                        except IndexError:
                            print("Index error!")
                        
                        ltubelets = np.delete(ltubelets, idx, axis=0)
                    else:
                        if offset >= K:
                            finished.append(it)

                # finished tubes that are done
                for it in finished[::-1]:  # process in reverse order to delete them with the right index why --++--
                    FINISHED_TUBES.append(CURRENT_TUBES[it][:])
                    del CURRENT_TUBES[it]

                # start new tubes
                for i in range(ltubelets.shape[0]):
                    CURRENT_TUBES.append([(frame, ltubelets[i, :])])

            # all tubes are not finished
            FINISHED_TUBES += CURRENT_TUBES

            # build real tubes
            output = []
            for t in FINISHED_TUBES:
                score = tubescore(t)

                beginframe = t[0][0]
                endframe = t[-1][0] + K - 1
                length = endframe + 1 - beginframe

                # delete tubes with short duraton; adjust accordingly
                if length < 15:
                    continue

                # build final tubes by average the tubelets
                out = np.zeros((length, 5), dtype=np.float32) # 6
                out[:, 0] = np.arange(beginframe, endframe + 1)
                n_per_frame = np.zeros((length, 1), dtype=np.int32)
                for i in range(len(t)):
                    frame, box = t[i]
                    for k in range(K):
                        out[frame - beginframe + k, 1:5] += box[4 * k:4 * k + 4]
                        #out[frame - beginframe + k, -1] += box[-1]  # single frame confidence
                        n_per_frame[frame - beginframe + k, 0] += 1
                out[:, 1:] /= n_per_frame
                
                # pkl in MOC are integers (of type float32)
                out = out.astype(int).astype(np.float32)
                
                output.append(out) 
                # out: [num_frames, (frame idx, x1, y1, x2, y2)]
        
        
        '''
        ### debug: uncomment the block to visualize linked results
        
        # for each linked tube
        # output: list of list, containing array of bboxes and tube score 
        
        vlist_im = v + '/images/left/ev_inf/'
        all_files_im = os.listdir(vlist_im)
        all_files_im.sort()
        
        for idx, tb in enumerate(output):
            start_f = int(tb[0][0])
            end_f = int(tb[-1][0])
            
            for im in range(start_f, end_f+1):
                
                im_file = vlist_im + all_files_im[im-1] 
                im_data = plt.imread(im_file)
                height, width, nbands = im_data.shape
                dpi = 80
                figsize = width / float(dpi), height / float(dpi)
                
                fig = plt.figure(figsize=figsize)
                ax = fig.add_axes([0,0,1,1])
                ax.axis('off')
                ax.imshow(im_data, interpolation='nearest')
                
                
                x1, y1, x2, y2 = tb[im-start_f][1:5]
                edgecolor = 'yellow'
                
                ax.add_patch(plt.Rectangle((x1, y1), x2- x1, y2 - y1, fill=False, edgecolor=edgecolor, linewidth=3))
                
                text = 'Frame ' + str(im-1) 
                ax.text(x1 - 2, y1 - 10, text, bbox=dict(facecolor='navy', alpha=1.0), fontsize=14, color='yellow')
                plt.show()
        
        '''
        RES[ilabel] = output
        
        video_file = v_boxpath.split('/')[7]
        gttubes[video_file] = RES
        nframes_dict[video_file] = nframes
        resolution[video_file] = (480, 640)
        
    GT['gttubes'] = gttubes
    GT['nframes'] = nframes_dict
    GT['resolution'] = resolution
    
    # TODO: modify the output path to store pkl
    outfile = None
    if outfile is None:
        print('Modify the output path accordingly.')
        sys.exit()
        
    with open(outfile, 'wb') as fid:
        pickle.dump(GT, fid)
        
    print('Finished generating GT pkl.')
    
if __name__ == '__main__':
    
    
    ### 1. check gt format from JHMDB-21/UCF-24
    pkl_file = '/home/alphadadajuju/projects/TEDdet_clean/data/ucf24/UCF101v2-GT.pkl'

    with open(pkl_file, 'rb') as fid:
        pkl = pickle.load(fid, encoding='iso-8859-1')
    
    
    '''
    ### 1.5 check detection pkl
    det_path = '/home/alphadadajuju/projects/TEDdet_clean/data0/ted_K5_jh_s1_stream/brush_hair/April_09_brush_hair_u_nm_np1_ba_goo_0/00021.pkl'
    with open(det_path, 'rb') as fid:
                dets = pickle.load(fid)
    
    print('Load detection.')
    '''
    
    # to generate gt pkl from kitti
    #BuildGT()
    
    # uncomment to verify generated gt pkl
    pkl_file = '/home/alphadadajuju/projects/event_camera_judy/kitti-generated-GT.pkl'
    with open(pkl_file, 'rb') as fid:
        pkl_self = pickle.load(fid, encoding='iso-8859-1')
    
    print ('Loaded pkl.')
    
             