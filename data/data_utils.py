# -*- coding: utf-8 -*-
"""
Created on Wed Feb 10 16:49:20 2021

@author: Aline Sindel
"""

import numpy as np
import os

import decord
from decord import VideoReader


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in [".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".PNG", ".JPG", ".JPEG", ".BMP", ".TIF", ".TIFF"])

def is_text_file(filename):
    return any(filename.endswith(extension) for extension in [".csv", ".txt", ".json"])

def is_video_file(filename):
    return any(filename.endswith(extension) for extension in [".mp4", ".m4v", ".mov", ".avi"])

def crop_frame(frame, x1, y1, x2, y2):
    return frame[y1:y2, x1:x2]

def crop_frames(frames, x1, y1, x2, y2):
    return frames[:,y1:y2, x1:x2] 

def determine_load_size_roi(videofile, rois, patch_size, full_size=False):
    _,name = os.path.split(videofile)
    base,_ = os.path.splitext(name)

    #load size roi
    roi = rois[base] #roi: x1,y1,x2,y2
    #size of roi
    w = roi[2] - roi[0] #x2-x1
    #h = roi[3] - roi[1] #y2-y1
    #scaling factor
    f = patch_size/w            
    vr0 = VideoReader(videofile)            
    frame = vr0[0]
    H1,W1,_ = frame.shape #HR video: 1920x1080
    H2 = round(H1*f)
    W2 = round(W1*f)            
    load_size_roi = np.array((H2,W2), np.int32)
    #scale roi
    roi = np.round(roi*f).astype(np.int32)  
    
    if full_size:
        #load size full            
        f2 = patch_size/W1
        H3 = round(H1*f2)
        W3 = round(W1*f2)            
        load_size_full = np.array((H3,W3), np.int32) 
        return base, roi, load_size_roi, load_size_full
        
    return base, roi, load_size_roi

def read_labels(label_file):
    f = open(label_file, "r")
    roi_dict = dict()
    head = f.readline()
    for line in f:        
        line_split = line.split(',')
        if len(line_split)>2:
            file = line_split[0]
            roi_dict[file] = np.array(line_split[1:]).astype(np.int32)
    f.close()
    return roi_dict
    
def read_pred_slide_ids_from_file(file):
    f = open(file, "r")  
    h = f.readline()
    slide_ids = []
    frame_ids_1 = []
    frame_ids_2 = []
    for line in f:
        line_split = line.split(", ")
        slide_id = int(np.float(line_split[0]))
        slide_ids.append(slide_id)        
        frame_id_1 = int(np.float(line_split[1]))
        frame_ids_1.append(frame_id_1)
        frame_id_2 = int(np.float(line_split[2]))
        frame_ids_2.append(frame_id_2)          
    f.close()   
    return np.array(slide_ids), np.array(frame_ids_1), np.array(frame_ids_2)
   
    
def extract_slide_transitions(slide_ids, frame_ids_1, frame_ids_2):   
    slide_transition_pairs = np.vstack((frame_ids_2[:-1],frame_ids_1[1:]))
    frame_types = np.vstack((slide_ids[:-1]<0,slide_ids[1:]<0)).astype(np.uint8)
    slide_transition_types = slide_transition_pairs[1] - slide_transition_pairs[0]
    slide_transition_types = (slide_transition_types>1).astype(np.uint8) #0: hard transition, 1: gradual transition    
    return slide_transition_pairs.transpose([ 1, 0]), frame_types.transpose([ 1, 0]), slide_transition_types


       
