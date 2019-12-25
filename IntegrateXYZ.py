#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 22 10:33:29 2019

@author: goutham
"""

import numpy as np
from plyfile import PlyData, PlyElement
from PIL import Image
import math
import pandas as pd 
import pyrealsense2 as rs
import os

prediction_dir='C:\Users\gouth\OneDrive\Documents\senior_design/2ratsIRbagDLC_resnet50_2rats_finalNov10shuffle1_10000.csv'
out_dir='C:\Users\gouth\OneDrive\Documents\senior_design\depth_frames'

def get_xy(count,prediction_data):
    result_for_frame_x=[]
    result_for_frame_y=[]
    [rows,columns]=prediction_data.shape
    for j in range(0,int(columns/2)):
        x_true=prediction_data.iat[count,2*j]
        y_true=prediction_data.iat[count,2*j+1]
        result_for_frame_x.append(x_true)
        result_for_frame_y.append(y_true)
        
    return result_for_frame_x,result_for_frame_y


prediction_data=pd.read_csv(prediction_dir,header=1)


prediction_data=pd.read_csv(prediction_dir,header=1)

prediction_data=prediction_data[['Snout 1','Snout 1.1','Neck 1','Neck 1.1','Tailbase 1','Tailbase 1.1','Front Paw 1',
        'Front Paw 1.1','Back Paw 1', 'Back Paw 1.1', 'Snout 2', 'Snout 2.1','Neck 2',
       'Neck 2.1','Tailbase 2', 'Tailbase 2.1','Front Paw 2','Front Paw 2.1', 'Back Paw 2',
       'Back Paw 2.1' ]]
# Declare pointcloud object, for calculating pointclouds and texture mappings
pc = rs.pointcloud()
# We want the points object to be persistent so we can display the last cloud when a frame drops
points = rs.points()
cfg = rs.config()
cfg.enable_device_from_file('rats2.bag')
# Declare RealSense pipeline, encapsulating the actual device and sensors
pipe = rs.pipeline()
#Start streaming with default recommended configuration
count=0
prof=pipe.start(cfg)
prof.get_device().as_playback().set_real_time(False)
depth_sensor = prof.get_device().first_depth_sensor()
depth_scale = depth_sensor.get_depth_scale()
[rows,columns]=prediction_data.shape
new_col=int(float(1.5*columns))
overall_df=np.zeros((rows,new_col))
while True:
    # Fetch color and depth frames
    count=count+1
    frames = pipe.wait_for_frames()
    depth = frames.get_depth_frame()
    color = frames.get_color_frame()
    depth_intrin = depth.profile.as_video_stream_profile().intrinsics
    # Generate the pointcloud and texture mappings
    #print(depth_intrin)
    points = pc.calculate(depth)
    [value_x,value_y]=get_xy(count,prediction_data)
    results=[]
    for i in range(0,len(value_x)):
        y=value_y[i]
        x=value_x[i]
        y=int(round(float(y)))
        x=int(round(float(x)))
        depth_pixel = [y,x]
        depth_image = np.asanyarray(depth.get_data())
        depth_value = depth_image[y][x]*depth_scale
        #depth_point = rs.rs2_deproject_pixel_to_point(depth_intrin, depth_pixel,depth_scale)
        depth_point = rs.rs2_deproject_pixel_to_point(depth_intrin,depth_pixel,depth_value)
        if abs(depth_point[0])==0 and abs(depth_point[1])==0 and abs(depth_point[2])==0:
            depth_pixel=[y+1,x+1]
            depth_value=depth_image[y+1][x+1]*depth_scale
            depth_point = rs.rs2_deproject_pixel_to_point(depth_intrin,depth_pixel,depth_value)
        if abs(depth_point[0])==0 and abs(depth_point[1])==0 and abs(depth_point[2])==0:
            depth_pixel=[y-1,x-1]
            depth_value=depth_image[y-1][x-1]*depth_scale
            depth_point = rs.rs2_deproject_pixel_to_point(depth_intrin,depth_pixel,depth_value)
        if abs(depth_point[0])==0 and abs(depth_point[1])==0 and abs(depth_point[2])==0:
            depth_pixel=[y+1,x]
            depth_value=depth_image[y+1][x]*depth_scale
            depth_point = rs.rs2_deproject_pixel_to_point(depth_intrin,depth_pixel,depth_value)
        if abs(depth_point[0])==0 and abs(depth_point[1])==0 and abs(depth_point[2])==0:
            depth_pixel=[y,x+1]
            depth_value=depth_image[y][x+1]*depth_scale
            depth_point = rs.rs2_deproject_pixel_to_point(depth_intrin,depth_pixel,depth_value)
            
        results.append(depth_point[0])
        results.append(depth_point[1])
        results.append(depth_point[2])
        
    overall_df[count,:]=np.asarray(results)
    if count==1000:
        break

pipe.stop()

final_dataframe=pd.DataFrame(overall_df)

final_dataframe.columns=['Snout 1 x','Snout 1 y','Snout 1 z','Neck 1 x','Neck 1 y','Neck 1 z','Tailbase 1 x','Tailbase 1 y','Tailbase 1 z','Front Paw 1 x','Front Paw 1 y','Front paw 1 z',
        'Back Paw 1 x','Back Paw 1 y', 'Back Paw 1 z', 'Snout 2 x','Snout 2 y','Snout 2 z','Neck 2 x','Neck 2 y','Neck 2 z','Tailbase 2 x','Tailbase 2 y','Tailbase 2 z','Front Paw 2 x','Front Paw 2 y','Front paw 2 z',
        'Back Paw 2 x','Back Paw 2 y', 'Back Paw 2 z' ]

final_dataframe.to_excel("xyz2_2rat_exp.xlsx") 











