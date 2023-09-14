# -*- coding: utf-8 -*-
"""
Created on Sun Aug 29 18:16:03 2021

Code is partly based on https://git.aweirdimagination.net/perelman/slide-detector

@author: Aline Sindel
"""

import os
import random
import argparse

import numpy as np
import cv2

import torch
import torch.nn as nn

#import decord
#from decord import VideoReader
from emily_helper_functions.video_reader import get_frames_as_tensor

from model import *

from data.data_utils import *
from data.test_video_clip_dataset import BasicTransform

def detect_initial_slide_transition_candidates_resnet2d(net, videofile, base, roi, load_size_roi, out_dir, opt):
    # load video file
    #vr = VideoReader(videofile, width=load_size_roi[1], height=load_size_roi[0]) 
    vr = get_frames_as_tensor(videofile, "MoviePy", 1)

    #determine number of frames
    N_frames = len(vr)  
    print(f"number of frames to be processed: {N_frames}")
    anchor_frame = None
    anchor_frame_idx = -1
    video_frame_idx = None
    prev_video_frame_idx = None
    slide_id = -1
    slide_ids = []
    frame_ids_1 = []
    frame_ids_2 = []
    
    if opt.in_gray:
        data_shape = "2_channel"
        opt.input_nc = 2
    else:
        data_shape = "6_channel"
        opt.input_nc = 6
    my_transform = BasicTransform(data_shape = data_shape) #, blur = opt.blur)
    activation = nn.Sigmoid()

    all_predictions = []
    
    for i in range(N_frames):

        frame = vr[i]

        # create imgs to store first anchor and second img to compare with. (2 frames, H, W, C) # 2 channels for gray
        imgs = torch.zeros((2,opt.patch_size,opt.patch_size,int(opt.input_nc/2)))
            
        if opt.in_gray: #opencv rgb2gray for torch
            print("debugging: opt.in_gray true", frame.shape)
            frame = 0.299*frame[...,0]+0.587*frame[...,1]+0.114*frame[...,2]
            frame = frame.unsqueeze(2) # but nothing changed!!
            print("after multiplications and unsqueeze, ", frame.shape)
        
        # crop to bounding box region
        frame = crop_frame(frame,roi[0],roi[1],roi[2],roi[3])  # frame.shape = torch.Size([136, 256, 1])
        #scale to max size (in case patch size changed)
        img_max_size = max(frame.shape[0], frame.shape[1]) # None, 256
        scaling_factor = opt.patch_size / img_max_size
        if scaling_factor != 1:         
            frame = cv2.resize(frame, (round(frame.shape[1] * scaling_factor), round(frame.shape[0] * scaling_factor)), interpolation = cv2.INTER_NEAREST)
            H,W,C = frame.shape
            imgs[1,:H,:W,:C] = frame
        else:
            H,W,C = frame.shape
            imgs[1,:H,:W,:C] = frame
        
        #set anchor
        if anchor_frame == None:
            imgs[0,:H,:W,:C] = frame
            anchor_frame_idx = i 
            anchor_frame = frame
        else:
            imgs[0,:H,:W,:C] = anchor_frame
            
        imgs = my_transform(imgs)  # permutes, reshapes, and normalises pixels -> ONE image of 4 layers instead of 2 image 2 layers (for gray)
        
        with torch.no_grad():
            #imgs = imgs.cuda()
             
            pred = net(imgs.unsqueeze(0))
            pred = pred.squeeze(1)            
            pred = activation(pred)
            all_predictions.append(pred)
            if pred<0.5: #transition (class 0)
                if (i - anchor_frame_idx) > opt.slide_thresh: #static frame
                    if video_frame_idx is not None: 
                        if (video_frame_idx - prev_video_frame_idx) > opt.video_thresh:
                            print("video frame {} at {} to {}".format(-1,prev_video_frame_idx+1, video_frame_idx+1))
                            slide_ids.append(-1)
                            frame_ids_1.append(prev_video_frame_idx+1)
                            frame_ids_2.append(video_frame_idx+1) 
                        video_frame_idx = None
                        prev_video_frame_idx = None
                    
                    slide_id += 1
                    print("static slide {} at {} to {}".format(slide_id,anchor_frame_idx+1, i))
                    slide_ids.append(slide_id)
                    frame_ids_1.append(anchor_frame_idx+1)
                    frame_ids_2.append(i)

                else:
                   #video frame or grad transition
                   video_frame_idx = anchor_frame_idx
                   if prev_video_frame_idx is None:
                       prev_video_frame_idx = anchor_frame_idx                  
                #update anchor
                anchor_frame_idx = i
                anchor_frame = frame 
               
    print("length of all predicted results", len(all_predictions)) 
    frame_ids_1 = np.array(frame_ids_1)
    frame_ids_2 = np.array(frame_ids_2)
    
    #write to file
    print("-- write results file")
    logfile_path = os.path.join(out_dir, base + "_results.txt")
    f = open(logfile_path, "w")
    f.write('Slide No, FrameID0, FrameID1\n')
    f.close()    

    for slide_id,frame_id_1,frame_id_2 in zip(slide_ids,frame_ids_1,frame_ids_2):     
        print("{}, {}, {}\n".format(slide_id,frame_id_1,frame_id_2))          
        f = open(logfile_path, "a")
        f.write("{}, {}, {}\n".format(slide_id,frame_id_1,frame_id_2))
        f.close() 

    print("-- write prediction file")
    logfile_path = os.path.join(out_dir, base + "_all_pred_results.txt")
    f = open(logfile_path, "w")
    f.write(str(all_predictions))
    f.close()   
            
def test_resnet2d(opt):
    torch.manual_seed(0)
    random.seed(0)

    if os.path.exists(opt.out_dir)==False:
        os.makedirs(opt.out_dir)
    
    ####### Create model
    # --------------------------------------------------------------- 
    net = define_resnet2d(opt)       
    #net = net.cuda()
    net = loadNetwork(net, opt.model_path, checkpoint=opt.load_checkpoint, prefix='')
    net.eval()

    #### Create dataloader
    # ---------------------------------------------------------------  
    video_dir = "videos/" + opt.phase   #opt.dataset_dir + "/videos/" + opt.phase   

    videoFilenames = []
    videoFilenames.extend(os.path.join(video_dir, x)
                                         for x in sorted(os.listdir(video_dir)) if is_video_file(x))
    
    roi_path = os.path.join("videos", opt.phase+'_bounding_box_list.txt') # os.path.join(opt.dataset_dir,"videos", opt.phase+'_bounding_box_list.txt')
    rois = read_labels(roi_path)

    #decord.bridge.set_bridge('torch')
    
    for k,videofile in enumerate(videoFilenames):
        print("Processing video No. {}: {}".format(k+1, videofile))
        
        base, roi, load_size_roi = determine_load_size_roi(videofile, rois, opt.patch_size)
         
        detect_initial_slide_transition_candidates_resnet2d(net, videofile, base, roi, load_size_roi, opt.out_dir, opt)

            
                  
if __name__ == '__main__':
    parser = argparse.ArgumentParser('slide_detection') 
    parser.add_argument('--dataset_dir', help='path to dataset dir',type=str, default='C:/Users/Sindel/Project/Data/datasets/LectureVideos') 
    parser.add_argument('--out_dir', help='path to dataset dir',type=str, default='C:/Users/Sindel/Project/Code/SliTraNet/results/test/resnet18_gray')
    parser.add_argument('--backbone_2D', help='name of backbone (resnet18 or resnet50)',type=str, default='resnet18')   
    parser.add_argument('--model_path', help='path of weights',type=str, default='C:/Users/Sindel/Project/Code/SliTraNet/weights/Frame_similarity_ResNet18_gray.pth')
    parser.add_argument('--load_checkpoint', type=bool, default=False, help='model_path is path to checkpoint (True) or path to state dict (False)')
    parser.add_argument('--slide_thresh', type=int, default=8, help='threshold for minimum static slide length')
    parser.add_argument('--video_thresh', type=int, default=13, help='threshold for minimum video length to distinguish from gradual transition')
    parser.add_argument('--patch_size', type=int, default=256, help='network input patch size')
    parser.add_argument('--n_class', type=int, default=1, help='number of classes')
    parser.add_argument('--input_nc', type=int, default=2, help='number of input channels for ResNet: gray:2, RGB:6')
    parser.add_argument('--in_gray', type=bool, default=True, help='run network with grayscale input, else RGB')
    parser.add_argument('--phase', type=str, default='test', help='train, val, test, etc')


    opt = parser.parse_args()  

    test_resnet2d(opt)   