#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 19 19:14:47 2021

@author: dinesh
"""
import glob
from .utils import bb_intersection_over_union_unoccluded
import numpy as np
from PIL import Image
import datetime
import cv2
import os
from tqdm import tqdm


def get_image(time, folder):
    for week_loop in range(5):
        try:
            image = np.array(Image.open(folder+'/week' +str(week_loop)+'/'+ str(time).replace(' ','T').replace(':','-').split('+')[0] + '.jpg'))                
            break
        except:
            continue
    if image is None:
        print('file not found')
    return image

def get_mask(segm, image):
    poly = np.array(segm).reshape((int(len(segm)/2), 2))
    mask = image.copy()*0
    cv2.fillConvexPoly(mask, poly, (255, 255, 255))
    return mask

def get_unoccluded(indices, tracks_all):
    unoccluded_indexes = []
    unoccluded_index_all =[]
    while 1:
        unoccluded_clusters = []
        len_unocc = len(unoccluded_indexes)
        for ind in indices:
            if ind in unoccluded_indexes:
                continue
            occ = False
            for ind_compare in indices:
                if ind_compare in unoccluded_indexes:
                    continue
                if bb_intersection_over_union_unoccluded(tracks_all[ind], tracks_all[ind_compare]) > 0.01 and ind_compare != ind:
                    occ = True
            if occ==False:
                unoccluded_indexes.extend([ind])
                unoccluded_clusters.extend([ind])
        if len(unoccluded_indexes) == len_unocc and len_unocc != 0:
            for ind in indices:
                if ind not in unoccluded_indexes:
                    unoccluded_indexes.extend([ind])
                    unoccluded_clusters.extend([ind])
            
        unoccluded_index_all.append(unoccluded_clusters)
        if len(unoccluded_indexes) > len(indices)-5:
            break
    return unoccluded_index_all

def primes(n): # simple sieve of multiples 
   odds = range(3, n+1, 2)
   sieve = set(sum([list(range(q*q, n+1, q+q)) for q in odds], []))
   return [2] + [p for p in odds if p not in sieve]

def save_image(image_read, save_path, data, path):
        tracks = data['tracks_all_unoccluded']
        segmentations = data['segmentation_all_unoccluded']
        timestamps = data['timestamps_final_unoccluded']

        image = image_read.copy()
        indices = np.random.randint(len(tracks),size=30)
        prime_numbers = primes(1000)
        unoccluded_index_all = get_unoccluded(indices, tracks)
        
        mask_stacked = image*0
        mask_stacked_all =[]
        count = 0
        time = datetime.datetime.now()

        for l in indices:
                try:
                    image_crop = get_image(timestamps[l], path)
                except:
                    continue
                try:
                    bb_left, bb_top, bb_width, bb_height, confidence = tracks[l]
                except:
                    bb_left, bb_top, bb_width, bb_height, confidence, track_id = tracks[l]
                mask = get_mask(segmentations[l], image)
                
                image[mask > 0] = image_crop[mask > 0]
                mask[mask > 0] = 1
                for count, mask_inc in enumerate(mask_stacked_all):
                    mask_stacked_all[count][cv2.bitwise_and(mask, mask_inc) > 0] = 2
                mask_stacked_all.append(mask)
                mask_stacked += mask
                count = count+1
        
        cv2.imwrite(save_path + '/images/'+str(time).replace(' ','T').replace(':','-').split('+')[0] + '.jpg', image[:, :, ::-1])
        cv2.imwrite(save_path + '/Segmentation/'+str(time).replace(' ','T').replace(':','-').split('+')[0] + '.jpg', mask_stacked[:, :, ::-1]*30)
        np.savez_compressed(save_path+'/Segmentation/'+str(time).replace(' ','T').replace(':','-').split('+')[0], mask=mask_stacked_all)
        
def CWALT_Generation(camera_name):
    save_path_train = 'data/cwalt_train'
    save_path_test = 'data/cwalt_test'
    
    json_file_path = 'data/{}/{}.json'.format(camera_name,camera_name) # iii1/iii1_7_test.json' # './data.json'
    path = 'data/' + camera_name
    
    data = np.load(json_file_path + '.npz', allow_pickle=True)
    
    ## slip data
    
    data_train=dict()
    data_test=dict()
    
    split_index = int(len(data['timestamps_final_unoccluded'])*0.8)
    
    data_train['tracks_all_unoccluded'] = data['tracks_all_unoccluded'][0:split_index]
    data_train['segmentation_all_unoccluded'] = data['segmentation_all_unoccluded'][0:split_index]
    data_train['timestamps_final_unoccluded'] = data['timestamps_final_unoccluded'][0:split_index]
    
    data_test['tracks_all_unoccluded'] = data['tracks_all_unoccluded'][split_index:]
    data_test['segmentation_all_unoccluded'] = data['segmentation_all_unoccluded'][split_index:]
    data_test['timestamps_final_unoccluded'] = data['timestamps_final_unoccluded'][split_index:]

    image_read = np.array(Image.open(path + '/T18-median_image.jpg'))
    image_read = cv2.resize(image_read, (int(image_read.shape[1]/2), int(image_read.shape[0]/2)))
    
    try:
        os.mkdir(save_path_train)
    except:
        print(save_path_train)

    try:
        os.mkdir(save_path_train + '/images')
        os.mkdir(save_path_train + '/Segmentation')
    except:
        print(save_path_train+ '/images')

    try:
        os.mkdir(save_path_test)
    except:
        print(save_path_test)

    try:
        os.mkdir(save_path_test + '/images')
        os.mkdir(save_path_test + '/Segmentation')
    except:
        print(save_path_test+ '/images')

    for loop in tqdm(range(3000), desc="Generating training CWALT Images "):
        save_image(image_read, save_path_train, data_train, path)
        
    for loop in tqdm(range(300), desc="Generating testing CWALT Images "):
        save_image(image_read, save_path_test, data_test, path)

