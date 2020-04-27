#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 13 19:50:36 2019

@author: goutham

## written with python 3

"""

import numpy as np
import open3d as o3d


pcd=o3d.io.read_point_cloud("/Users/goutham/Documents/Senior_year/research_design/1.ply")
nump_mat=np.asarray(pcd)

o3d.visualization.draw_geometries([pcd])



# future experimets will involve manipulating this numpy array to find the appropriate z coordinates for (x,y)



