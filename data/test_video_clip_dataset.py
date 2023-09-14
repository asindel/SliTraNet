# -*- coding: utf-8 -*-
"""
Created on Fri Apr 30 16:00:51 2021

@author: Aline Sindel
"""


import numpy as np

import torch
import torch.utils.data as data

import cv2
#import decord
#from decord import VideoReader

from data.data_utils import *


class BasicTransform():
    def __init__(self, data_shape = "CNHW"):
        self.data_shape = data_shape
        
    def __call__(self, imgs):
        """
        imgs: N,H,W,C uint8
        labels: k
        """
        if self.data_shape == "CNHW":
            #convert NHWC to CNHW
            imgs = imgs.permute(3, 0, 1, 2)
        elif self.data_shape == "6_channel" or self.data_shape == "2_channel":
            #convert NHWC to NCHW    
            imgs = imgs.permute(0, 3, 1, 2)
            N,C,H,W = imgs.shape
            imgs = imgs.reshape(N*C,H,W)
        
        #normalize to 0 1
        imgs = imgs.float()/255 
           
        imgs -= 0.5
      
        return imgs


class VideoClipTestDataset(data.Dataset):
    def __init__(self, videofile, load_size, slide_transition_pairs, patch_size, clip_length, temporal_sampling, n_channels = 3, transform = None, roi = None):
        self.videofile = videofile
        self.load_size = load_size

        self.patch_size = patch_size 
        self.clip_length = clip_length
        self.temporal_sampling = temporal_sampling
        self.n_channels = n_channels        

        #decord.bridge.set_bridge('torch')

        clip_list = []
        transition_no = []
        #self.vr = VideoReader(videofile, width=self.load_size[1], height=self.load_size[0])
        self.vr = get_frames_as_tensor(videofile, "MoviePy")
        self.roi = roi
        self.transform = transform

        #slide transitions
        half_clip_length = int(0.5*self.clip_length)
        for i,pair in enumerate(slide_transition_pairs):
            diff = pair[1] - pair[0]
            if diff <= 1: #hard transition or no transition or uncertain
                clip_ids = np.array(range(pair[0]-half_clip_length*self.temporal_sampling, pair[0]+half_clip_length*self.temporal_sampling, self.temporal_sampling))
                clip_list.append(clip_ids)
                transition_no.append(i)
            elif (diff <= self.clip_length): #gradual or video or uncertain  
                mid = pair[0] + int(0.5*diff) 
                clip_ids = np.array(range(mid-half_clip_length*self.temporal_sampling, mid+half_clip_length*self.temporal_sampling, self.temporal_sampling))
                clip_list.append(clip_ids)
                transition_no.append(i)                
                clip_ids = np.array(range(pair[0]-half_clip_length*self.temporal_sampling, pair[0]+half_clip_length*self.temporal_sampling, self.temporal_sampling))
                clip_list.append(clip_ids)
                transition_no.append(i)
                clip_ids = np.array(range(pair[1]-half_clip_length*self.temporal_sampling, pair[1]+half_clip_length*self.temporal_sampling, self.temporal_sampling))
                clip_list.append(clip_ids)
                transition_no.append(i)
            else: #gradual or video or uncertain  
                k = int(diff/(half_clip_length))+1
                ps = np.linspace(pair[0], pair[1], num=k).astype(np.int32)
                for p in ps:
                    clip_ids = np.array(range(p-half_clip_length*self.temporal_sampling, p+half_clip_length*self.temporal_sampling, self.temporal_sampling))
                    clip_list.append(clip_ids)
                    transition_no.append(i) 

        self.clip_list = np.vstack(clip_list) 
        self.transition_no_list = np.vstack(transition_no) 
  

        self.N_clips = len(self.clip_list) 
        print("Dataset size:", self.N_clips)
            
    def __getitem__(self, index):
        
        clip_files = self.clip_list[index] #  returns [24 25 26 27 28 29 30 31], and 31 other clip_files !! 
        transition_no = self.transition_no_list[index]
        print("clip files:", clip_files, "transition_no", transition_no)
        #read frames        
        frames = torch.stack(tuple(self.vr[indx] for indx in clip_files), 0) # removed .to('cpu') #np.array([self.vr[indx].to('cpu') for indx in clip_files]) #.get_batch(clip_files)
        # crop to bounding box region
        if self.roi is not None:
            frames = crop_frames(frames,self.roi[0],self.roi[1],self.roi[2],self.roi[3])   
        
        imgs = torch.zeros((self.clip_length,self.patch_size,self.patch_size,self.n_channels))
        
        #scale to max size (in case patch size changed)
        img_max_size = max(frames.shape[1], frames.shape[2])
        scaling_factor = self.patch_size / img_max_size
        if scaling_factor != 1:            
            for i,img in enumerate(frames):
                img = cv2.resize(img.numpy(), (round(img.shape[1] * scaling_factor), round(img.shape[0] * scaling_factor)), interpolation = cv2.INTER_NEAREST)
                H,W,C = img.shape
                imgs[i,:H,:W,:C] = torch.from_numpy(img)
        else:
            N,H,W,C = frames.shape
            imgs[:,:H,:W,:C] = frames
                    
        if self.transform is not None:
            imgs = self.transform(imgs)
            
        return imgs, clip_files, transition_no


    def __len__(self):
        return self.N_clips 
    
