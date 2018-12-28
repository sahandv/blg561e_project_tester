#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Istanbul Technical University - Computer Engineering

BLG 561E Project Test Code

Created on Wed Dec 19 11:09:01 2018

"""

from imageio import imread
import xmltodict as xd
import numpy as np
import pandas as pd
import os
import time

# =============================================================================
#    In order to reduce the file loading and itearation overhead, and also
#    excluding other processes from time calculations, you will be asked to
#    provide predict function in your code. This will make the competition fair 
#    for everyone.
#    In this script, we will import your predict function and initialization 
#    block for your model.
#
#    This function should:
#    * Get the frame data (make sure tu use RGB and not BGR in your training. 
#      otherwise, convert your images to BGR from RGB in predict function.)
#    * Return detection result as a list of object dictionaries. (Similar to 
#      the sample results provided. Coordinates as integer and name as string.)
#
#    (Remember that you have to provide the predict function and your trained
#    model yourself. This predict function is only for demonstration purpose. )
# =============================================================================

# =============================================================================
# INITIALIZATION BLOCK FOR YOUR MODEL:
# =============================================================================
"""
Write your initialization code here. you may import your model and weights 
and everything here.
"""

# =============================================================================
# SAMPLE predict() FUNCTION:
# =============================================================================
def predict(frame_img):
    result = [{'bndbox':{'xmax':1170,'xmin':1130,'ymax':1894,'ymin':1823},
                         'name':'33','confidence':0.3},
              {'bndbox':{'xmax':2704,'xmin':2665,'ymax':1731,'ymin':1710},
                         'name':'33','confidence':0.9},
              {'bndbox':{'xmax':2504,'xmin':2495,'ymax':1731,'ymin':1710},
                         'name':'33','confidence':0.9},
              {'bndbox':{'xmax':1704,'xmin':1665,'ymax':1831,'ymin':1810},
                         'name':'34','confidence':0.9},
              {'bndbox':{'xmax':2704,'xmin':2665,'ymax':1731,'ymin':1710},
                         'name':'33','confidence':0.99},
              {'bndbox':{'xmax':2704,'xmin':2665,'ymax':1731,'ymin':1710},
                         'name':'36','confidence':0.99},
              {'bndbox':{'xmax':1170,'xmin':1130,'ymax':1894,'ymin':1823},
                         'name':'33','confidence':0.83}
    ]
    return result

# =============================================================================
# Test Code Start
# =============================================================================
    
def iou_comp(bbx_a,bbx_b):
    assert bbx_a['xmin'] < bbx_a['xmax']
    assert bbx_a['ymin'] < bbx_a['ymax']
    assert bbx_b['xmin'] < bbx_b['xmax']
    assert bbx_b['ymin'] < bbx_b['ymax']
    x_left = max(bbx_a['xmin'], bbx_b['xmin'])
    y_top = max(bbx_a['ymin'], bbx_b['ymin'])
    x_right = min(bbx_a['xmax'], bbx_b['xmax'])
    y_bottom = min(bbx_a['ymax'], bbx_b['ymax'])
    if x_right < x_left or y_bottom < y_top:
        return 0.0
    intersection_area = (x_right - x_left) * (y_bottom - y_top)
    bbx_a_area = (bbx_a['xmax'] - bbx_a['xmin']) * (bbx_a['ymax'] - bbx_a['ymin'])
    bbx_b_area = (bbx_b['xmax'] - bbx_b['xmin']) * (bbx_b['ymax'] - bbx_b['ymin'])
    iou = intersection_area / float(bbx_a_area + bbx_b_area - intersection_area)
    assert iou >= 0.0
    assert iou <= 1.0
    
    return iou

# =============================================================================
# Preparation
# =============================================================================
source_xml_dir = 'sample_data/annotations/'
source_img_dir = 'sample_data/JPEGImages/'
frame_times = []
frame_APs = []
iou_thresh = 0.30
# =============================================================================
# Iterate over files
# =============================================================================
for root, dirs, files in os.walk(source_xml_dir):
    for file in files:
        if file.endswith(".xml"):
            xml_file = os.path.join(root, file)
            img_file = source_img_dir+str.split(file,'.')[0]+'.jpg'
# =============================================================================
#             Read GT data
# =============================================================================
            frame = imread(img_file)
            tree_root = None
            with open(xml_file,'rb') as f:
                tree_root = xd.parse(f)
            objects_gt = tree_root['annotation']['object']
            for object_gt in objects_gt:
                object_gt['bndbox']['xmax'] = int(object_gt['bndbox']['xmax'])
                object_gt['bndbox']['xmin'] = int(object_gt['bndbox']['xmin'])
                object_gt['bndbox']['ymax'] = int(object_gt['bndbox']['ymax'])
                object_gt['bndbox']['ymin'] = int(object_gt['bndbox']['ymin'])
# =============================================================================
#             Prediction run
# =============================================================================
            start_time = time.time()
            objects_pred = predict(frame)
            frame_times.append(time.time() - start_time)
# =============================================================================
#             Result vs. GT comparison
# =============================================================================
            object_union = []
            object_pred_array = []
            TP = []
            FP = []
            
            for i_gt,object_gt in enumerate(objects_gt):
                for i_pred,object_pred in enumerate(objects_pred):
                    # IOU computation   
                    iou = iou_comp(object_pred['bndbox'],object_gt['bndbox'])
                    if iou > iou_thresh:
                        if object_gt['name']==object_pred['name']:
                            TP.append([i_gt,i_pred,iou,object_pred['confidence']])
            
                            
            # TP refine
            TP = np.array(TP)
            if TP.shape[0]>0:
                TP = TP[np.lexsort((-TP[:,3],-TP[:,2]))]
            
            TP_mask = []
            visited_gt = []
            visited_pred = []
            for key,element in enumerate(TP):
                if (element[0] in visited_gt):# or (element[1] in visited_pred):
                    TP_mask.append(False)
                else:
                    visited_gt.append(element[0])
                    visited_pred.append(element[1])
                    TP_mask.append(True)
            TP = TP[TP_mask]
            
            # FP count
            for i_pred,object_pred in enumerate(objects_pred):
                is_TP = 0
                if TP.shape[0]>0:
                    if i_pred in TP[:,1]:
                        is_TP = 1
                object_pred_array.append([i_pred,object_pred['confidence'],is_TP,0,0]) # predID,confidence,TP,percision,recall
#                if i_pred not in TP[:,1]:
#                    FP.append([i_pred,object_pred['confidence']])
#            FP = np.array(FP)
            
            # FN count
#            for i_gt,object_gt in enumerate(objects_gt):
#                if i_gt not in TP[:,0]:
#                    FN.append([i_gt])
#            FN = np.array(FN)
            
            del visited_pred
            del visited_gt
            del TP_mask
            
# =============================================================================
#            Compute AP
# =============================================================================
            # For all classes, make a sorted list of percision and recall
            object_pred_array = np.array(object_pred_array)
            object_pred_array = object_pred_array[object_pred_array[:,1].argsort()[::-1]]
            
            all_possible_positives = len(object_gt) #TP.shape[0]
            accumulated_TP = 0
            
            for key,item in enumerate(object_pred_array):
                if item[2]==1:
                    accumulated_TP=accumulated_TP+1

                percision = accumulated_TP/(key+1)
                recall = accumulated_TP/all_possible_positives
                object_pred_array[key,3] = percision
                object_pred_array[key,4] = recall
                
            # Generate the max percision list (based on max recall)
            buffer_index = 0
            max_percision_val = 0
            max_percision = np.zeros((11,2),np.float64)
            key = 0
            for recall in range(0,11):
                recall = recall/10
                max_percision[key,1] = recall
                for item in object_pred_array:
                    # Find the last row having (a recall)<=(the recall)
                    if item[4]>recall:
                        break
                    if item[3]>max_percision_val:
                        max_percision_val = item[3]
                max_percision[key,0] = max_percision_val
                key = key+1
                
            AP = sum(max_percision[:,0])/11
            frame_APs.append(AP)

# =============================================================================
# Average FPS and AP calculator
# =============================================================================
total_time = float(sum(frame_times))
total_frames = float(len(frame_times))
average_FPS = float(total_frames/total_time)
average_AP = float(sum(frame_APs)/total_frames)

print('mean FPS',average_FPS)
print('mean AP',average_AP)

