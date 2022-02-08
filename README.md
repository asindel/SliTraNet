# SliTraNet
Automatic Detection of Slide Transitions in Lecture Videos using Convolutional Neural Networks

This is the source code to the conference article "SliTraNet: Automatic Detection of Slide Transitions in Lecture Videos using Convolutional Neural Networks" accepted at OAGM Workshop 2021.

If you use the code, please cite our paper (link will be added soon).

## Requirements

Install the requirements using pip or conda (python 3):
- torch >= 1.7
- torchvision
- opencv-contrib-python-headless
- numpy
- decord

## Usage

### Data

The dataset needs to be in the following folder structure:
- Video files in: "/videos/PHASE/", where PHASE is "train", "val" or "test".
- Bounding box labels in: "/videos/PHASE_bounding_box_list.txt"

Bounding box labels define the rectangle of the slide area in the format: Videoname,x0,y0,x1,y1

Here one example test_bounding_box_list.txt file (the header needs to be included):  
Video,x0,y0,x1,y1  
Architectures_1,38,57,1306,1008  
Architectures_2,38,57,1306,1008  


### Pretrained weights

The pretrained weights of SliTraNet from the paper can be downloaded [here](https://drive.google.com/drive/folders/1aQDVplbbpt-zgH2O1q7685AZ1hl0BsVV?usp=sharing).
Move them into the folder: "/weights"

### SliTraNet Inference: 

Run test_SliTraNet.py 

Some settings have to be specified, as described in the python file, such as the dataset and output folders and model paths.

Stage 1 of SliTraNet can also be applied separately (see test_slide_detection_2d.py) and afterwards the results can be loaded in test_SliTraNet.py.


@author Aline Sindel
