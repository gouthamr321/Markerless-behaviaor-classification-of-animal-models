# -*- coding: utf-8 -*-
"""
Created on Sun Oct 13 11:03:21 2019

@author: gouth

# written with python 2
"""
#We are trying to extract the pointclouds for each frame to a ply file


# First import the library
import pyrealsense2 as rs

# Declare pointcloud object, for calculating pointclouds and texture mappings
pc = rs.pointcloud()
# We want the points object to be persistent so we can display the last cloud when a frame drops
points = rs.points()
cfg = rs.config()
cfg.enable_device_from_file('rats2.bag')
# Declare RealSense pipeline, encapsulating the actual device and sensors
pipe = rs.pipeline()
#Start streaming with default recommended configuration
pipe.start(cfg)

try:
    
    frames = pipe.wait_for_frames()

    # Fetch color and depth frames
    depth = frames.get_depth_frame()
    color = frames.get_color_frame()

    # Generate the pointcloud and texture mappings
    points = pc.calculate(depth)

    points.export_to_ply("1.ply", color)
    print('Done')
    
finally:
    pipe.stop()