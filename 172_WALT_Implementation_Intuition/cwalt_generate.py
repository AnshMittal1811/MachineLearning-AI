#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun  4 16:55:58 2022

@author: dinesh
"""
from cwalt.CWALT import CWALT_Generation
from cwalt.Clip_WALT_Generate import Get_unoccluded_objects

if __name__ == '__main__':
    camera_name = 'cam2'
    Get_unoccluded_objects(camera_name)
    CWALT_Generation(camera_name)
