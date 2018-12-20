#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Istanbul Technical University - Computer Engineering

BLG 561E Project Test Code

Created on Wed Dec 19 11:09:01 2018

"""

from imageio import imread
import xmltodict as xd
import os
import time

# =============================================================================
#    In order to reduce the file loading and itearation overhead, and also
#    excluding other processes from time calculations, you will be asked to
#    provide predict function in your code. This will make the competition fair 
#    for everyone.
#    In this script, we will import your predict function.
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
def predict(frame_img):
    result = [{'bndbox':{'xmax':1170,'xmin':1130,'ymax':1894,'ymin':1823},
                         'name':'33'},
              {'bndbox':{'xmax':2704,'xmin':2665,'ymax':1731,'ymin':1710},
                         'name':'33'}
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
# Prepare files
# =============================================================================
source_xml_dir = 'sample_data/annotations/'
source_img_dir = 'sample_data/JPEGImages/'
frame_times = []
frame_APs = []
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
            comparison_results = []
            FN = 0
            TP = 0
            FP = 0
            # IOU computation
            for i_gt,object_gt in enumerate(objects_gt):
                for i_pred,object_pred in enumerate(objects_pred):
                    if object_gt['name']==object_pred['name']:
                        iou = iou_comp(object_gt['bndbox'],object_pred['bndbox'])
                        if iou > 0:
                            comparison_tmp = {'class_gt':object_gt['name'],
                                              'class_pred':object_pred['name'],
                                              'index_gt':i_gt,
                                              'index_pred':i_pred,
                                              'iou':iou}
                            comparison_results.append(comparison_tmp)
            # FNs
            detected_gts = []
            for intersection in comparison_results:
                detected_gts.append(intersection['index_gt'])
            expected_gts = list(range(len(objects_gt)))
            FN = len(objects_gt)-len(set(expected_gts) & set(detected_gts))
            # TN
            
            # TP
            
            # TBC ...


        

# =============================================================================
# Average FPS calulator
# =============================================================================
total_time = float(sum(frame_times))
total_frames = float(len(frame_times))
average_FPS = float(total_frames/total_time)



