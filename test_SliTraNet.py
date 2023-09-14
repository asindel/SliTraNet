# -*- coding: utf-8 -*-
"""
Created on Sun Aug 29 18:16:03 2021

@author: Aline Sindel
"""

import os
import random
import argparse
import numpy as np
import sys

import torch
import torch.nn as nn

#import decord
#from decord import VideoReader
from emily_helper_functions.video_reader import get_frames_as_tensor

from model import *
from test_slide_detection_2d import detect_initial_slide_transition_candidates_resnet2d

from data.data_utils import *
from data.test_video_clip_dataset import BasicTransform, VideoClipTestDataset


def printLog(*args, **kwargs):
    print(*args, **kwargs)
    with open('output.out','a') as file:
        print(*args, **kwargs, file=file)

def detect_slide_transitions(pred_feat):
    activation = nn.Softmax(dim=1)
    pred_labels = activation(pred_feat)
    scores, pred_classes = torch.max(pred_labels, 1)
    return pred_classes, scores


def test_SliTraNet(opt):
    torch.manual_seed(0)
    random.seed(0)

    if os.path.exists(opt.out_dir)==False:
        os.makedirs(opt.out_dir)
    
    ####### Create model
    # ---------------------------------------------------------------
    opt.n_class = 1
    net2d = define_resnet2d(opt)
    #net2d = net2d.cuda()
    net2d = loadNetwork(net2d, opt.model_path_2D, checkpoint=opt.load_checkpoint, prefix='')
    net2d.eval()    
    
    opt.n_class = 3
    net1 = ResNet3d(opt)
    #net1 = net1.cuda()
    net1 = loadNetwork(net1, opt.model_path_1, checkpoint=opt.load_checkpoint, prefix='module.')
    net1.eval() 

    opt.n_class = 4
    net2 = ResNet3d(opt)
    #net2 = net2.cuda()
    net2 = loadNetwork(net2, opt.model_path_2, checkpoint=opt.load_checkpoint, prefix='module.')
    net2.eval() 


    #### Create dataloader
    # --------------------------------------------------------------- 
    video_dir = "../videos/" + opt.phase  # opt.dataset_dir + "/videos/" + opt.phase   

    videoFilenames = []
    videoFilenames.extend(os.path.join(video_dir, x)
                                         for x in sorted(os.listdir(video_dir)) if is_video_file(x))
    print("videoFilenames:", videoFilenames)
    roi_path = os.path.join("../videos", opt.phase+'_bounding_box_list.txt') # os.path.join(opt.dataset_dir,"videos", opt.phase+'_bounding_box_list.txt')
    rois = read_labels(roi_path)
    print("Bounding Box labels READ:", rois)
    #decord.bridge.set_bridge('torch')
    
    for k,videofile in enumerate(videoFilenames):
        print("--- Processing video No. {}: {} ---".format(k+1, videofile))
        
        base, roi, load_size_roi, load_size_full = determine_load_size_roi(videofile, rois, opt.patch_size, full_size=True)

        ##################################################################
        ##  Stage 1: detect initial slide-slide and slide-video candidates  ##
        ##################################################################        
        
        print("--- start stage 1 ---")
        predfile = os.path.join(opt.pred_dir,base+'_results.txt')
        
        if os.path.exists(predfile)==False:
            # run stage 1
            if os.path.exists(opt.pred_dir)==False:
                os.makedirs(opt.pred_dir)            
        detect_initial_slide_transition_candidates_resnet2d(net2d, videofile, base, roi, load_size_roi, opt.pred_dir, opt)
            #### prints if the slides are static or not at this stage for each frmae!!! 
        # load results of stage 1
        print("--- loading results of stage 1 ---")
        slide_ids, slide_frame_ids_1, slide_frame_ids_2 = read_pred_slide_ids_from_file(predfile)
        slide_transition_pairs, frame_types, slide_transition_types = extract_slide_transitions(slide_ids, slide_frame_ids_1, slide_frame_ids_2)
        print(slide_ids, slide_frame_ids_1, slide_frame_ids_2)
        print("slide_transition_pairs:", slide_transition_pairs)
        ##################################################################
        ##  Stage 2: check slide - video candidates                     ##
        ##################################################################
        print("--- start stage 2 ---")
        full_clip_dataset = VideoClipTestDataset(videofile, 
                                            load_size_full,                                            
                                            slide_transition_pairs, 
                                            opt.patch_size, 
                                            opt.clip_length, 
                                            opt.temporal_sampling ,
                                            n_channels = 3, 
                                            transform = BasicTransform(data_shape = "CNHW")
                                            )
        full_clip_loader = torch.utils.data.DataLoader(full_clip_dataset, batch_size=opt.batch_size, shuffle=False, num_workers=0)
        
        slide_video_prediction = dict()

        with torch.no_grad():
            for j1, (clips, clip_inds, clip_transition_nums) in enumerate(full_clip_loader):
                #clips = clips.cuda()
                print(f"stage 2: clips is currently at {clips.get_device()} before net1")
                #torch.set_default_device("cpu") # avoid runtime errors on mpu
                pred1 = net1(clips) 

                #extract ids for slide transition candidates
                pred_classes, scores = detect_slide_transitions(pred1.squeeze(2).squeeze(2).detach().cpu())
                
                transition_nums = torch.unique(clip_transition_nums) #.cpu().numpy()
                print(transition_nums)
                for transition_no in transition_nums:
                    key = transition_no.numpy().tolist()
                    if key not in slide_video_prediction:
                        slide_video_prediction[key] = []
                        slide_video_prediction[key].append(pred_classes[torch.where(clip_transition_nums==transition_no)[0]].numpy())

            
            ##################################################################
            ##  Stage 3: check slide transition candidates                ##
            ##################################################################
            print("--- start stage 3 ---")
            clip_dataset = VideoClipTestDataset(videofile, 
                                                load_size_roi,                                            
                                                slide_transition_pairs, 
                                                opt.patch_size, 
                                                opt.clip_length, 
                                                opt.temporal_sampling ,
                                                n_channels = 3, 
                                                transform = BasicTransform(data_shape = "CNHW"),
                                                roi = roi)
            clip_loader = torch.utils.data.DataLoader(clip_dataset, batch_size=opt.batch_size, shuffle=False, num_workers=0)
            print("debugging and understanding: ", type(clip_dataset), type(clip_loader))
            slide_transition_prediction = dict()
            
            for j1, (clips, clip_inds, clip_transition_nums) in enumerate(clip_loader):
                #clips = clips.cuda()
                print("enumerating clip_loader:", j1, clip_inds, clip_transition_nums)
                pred2 = net2(clips)
                
                #extract ids for slide transition candidates
                pred_classes, scores = detect_slide_transitions(pred2.squeeze(2).squeeze(2).detach().cpu())
                
                transition_nums = torch.unique(clip_transition_nums)
                for transition_no in transition_nums:
                    key = transition_no.numpy().tolist()
                    if key not in slide_transition_prediction:
                        slide_transition_prediction[key] = []
                        slide_transition_prediction[key].append(pred_classes[torch.where(clip_transition_nums==transition_no)[0]].numpy())                

            logfile_path = os.path.join(opt.out_dir, base + "_transitions.txt")
            f = open(logfile_path, "w")
            f.write('Transition No, FrameID0, FrameID1\n')
            f.close() 
            
            s=0
            neg_indices = []
            for key in slide_transition_prediction.keys():
                slide_transition_pred = np.hstack(slide_transition_prediction[key])
                slide_video_pred = np.hstack(slide_video_prediction[key])

                if all(slide_transition_pred==3) and all(slide_video_pred==2):
                    neg_indices.append(key)

                else:
                    printLog("Pair:", slide_transition_pairs[key])
                    printLog("0:Hard transition, 1:grad transition, 2:slide, 3:video:", slide_transition_pred)
                    printLog("0:Transition, 1:slide, 2:video:", slide_video_pred) 

                    s+=1
                    pair = slide_transition_pairs[key]
                    f = open(logfile_path, "a")
                    f.write("{}, {}, {}\n".format(s,int(pair[0])+1,int(pair[1])+1))
                    f.close() 
            neg_indices = np.hstack(neg_indices)    
            mask = np.ones(len(slide_transition_pairs), dtype=bool)
            mask[neg_indices] = False
            filtered_slide_transition_pairs = slide_transition_pairs[mask,...]               
                
            print("Done")                

            
                  
if __name__ == '__main__':
    from datetime import datetime
    print(datetime.now())
    torch.set_default_device('cpu')
    ## remove pre-prend: C:/Users/Sindel/Project/Code/SliTraNet/
    parser = argparse.ArgumentParser('slide_detection') 
    parser.add_argument('--dataset_dir', help='path to dataset dir',type=str, default='../videos') 
    parser.add_argument('--out_dir', help='path to result dir',type=str, default='results/test/SliTraNet-gray-RGB')
    parser.add_argument('--patch_size', type=int, default=256, help='network input patch size')
    parser.add_argument('--phase', type=str, default='test', help='train, val, test, etc')
    parser.add_argument('--load_checkpoint', type=bool, default=False, help='model_path is path to checkpoint (True) or path to state dict (False)')
    ### Parameters for 2-D CNN
    parser.add_argument('--pred_dir', help='path to 2d result dir',type=str, default='results/test/resnet18_gray')
    parser.add_argument('--backbone_2D', help='name of 2d backbone (resnet18 or resnet50)',type=str, default='resnet18')   
    parser.add_argument('--model_path_2D', help='path of weights resnet2d',type=str, default='../weights/Frame_similarity_ResNet18_gray.pth')
    parser.add_argument('--slide_thresh', type=int, default=8, help='threshold for minimum static slide length')
    parser.add_argument('--video_thresh', type=int, default=13, help='threshold for minimum video length to distinguish from gradual transition') # change from 13
    parser.add_argument('--input_nc', type=int, default=2, help='number of input channels for ResNet: gray:2, RGB:6')
    parser.add_argument('--in_gray', type=bool, default=True, help='run resnet2d with grayscale input, else RGB')    
    ### Parameters for 3-D CNN
    parser.add_argument('--backbone_3D', help='name of 3d backbone (resnet18 or resnet50)',type=str, default='resnet50')
    parser.add_argument('--model_path_1', help='path of weights for 3D CNN Slide video detection',type=str, default='../weights/Slide_video_detection_3DResNet50.pth')
    parser.add_argument('--model_path_2', help='path of weights for 3D CNN Slide transition detection',type=str, default='../weights/Slide_transition_detection_3DResNet50.pth')
    parser.add_argument('--temporal_sampling', type=int, default=1, help='temporal sampling factor, e.g. 1: each frame, 5: each 5th frame')
    parser.add_argument('--clip_length', type=int, default=8, help='network input patch size')  
    parser.add_argument('--batch_size', type=int, default=32, help='batch size')   

    opt = parser.parse_args()  

    test_SliTraNet(opt)  
    print(datetime.now()) 