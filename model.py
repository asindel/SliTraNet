# -*- coding: utf-8 -*-
"""
Created on Fri Apr 30 16:00:51 2021

@author: Aline Sindel
"""

import torch
import torch.nn as nn
import os
from backbones import resnet2d, resnet3d

def define_resnet2d(cfg):
    """
    Network for Stage 1: Detection of initial slide-slide or slide-video transition candidates
    """    
    if cfg.backbone_2D == 'resnet18':
        net = resnet2d.resnet18(num_classes=cfg.n_class, input_nc=cfg.input_nc)   
    elif cfg.backbone_2D == 'resnet50':
        net = resnet2d.resnet50(num_classes=cfg.n_class, input_nc=cfg.input_nc) 
    else:    
        raise ValueError("Wrong backbone_2d model is requested. Please select it from [resnet18, resnet50]")     
    return net


class ResNet3d(nn.Module):
    """
    Network for Stage 2 and 3: Detection of slide transitions and video-slide transitions with 3d CNNN, Code of 3D backbone based on YOWO model, slightly modified (strides)
    """
    def __init__(self, cfg):
        super(ResNet3d, self).__init__()
        self.cfg = cfg

        ##### 3D Backbone #####
        if cfg.backbone_3D == "resnet18":
            self.backbone_3d = resnet3d.resnet18(shortcut_type='A')
            num_ch_3d = 512 # Number of output channels for backbone_3d
        elif cfg.backbone_3D== "resnet50":
            self.backbone_3d = resnet3d.resnet50(shortcut_type='B')
            num_ch_3d = 2048 # Number of output channels for backbone_3d
        else:
            raise ValueError("Wrong backbone_3d model is requested. Please select it from [resnet50, resnet18]")

        self.conv_final = nn.Conv2d(num_ch_3d, cfg.n_class, kernel_size=1, bias=False)

    def forward(self, x):
        print("check where tensor is in model.py forward function", x.get_device())
        x = self.backbone_3d(x)
        x = torch.squeeze(x, dim=2)
        out = self.conv_final(x)
        return out


def loadNetwork(net, model_path, checkpoint=True, prefix='module.'):
    print("Loading weights from:", model_path)
    if checkpoint:
        trained_checkpoint = torch.load(model_path, map_location='cpu')
        trained_state_dict = trained_checkpoint["state_dict"]
    else:
        trained_state_dict =  torch.load(model_path, map_location='cpu')
    
    my_state_dict = net.state_dict()
    for key in my_state_dict:
        module_key = prefix + key
        if module_key in trained_state_dict:
            weights = my_state_dict[key]
            pre_weights = trained_state_dict[module_key] 
            if weights.size() == pre_weights.size():
                my_state_dict[key] = pre_weights
            else:
                print("size mismatch for weight: ", key)                
        else:
            print("key: ", key)
    net.load_state_dict(my_state_dict, strict=True)  
    return net

