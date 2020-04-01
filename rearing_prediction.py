#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 28 21:13:32 2020

@author: goutham
"""

import numpy as np
from plyfile import PlyData, PlyElement
from PIL import Image
import math
import pandas as pd 
from sklearn import svm
import os
from sklearn.metrics import zero_one_loss
from sklearn.metrics import confusion_matrix

def Euclidean_distance(x_1,y_1,x_2,y_2):

    euc_dist=np.sqrt((((x_1-x_2)**2) + ((y_1-y_2)**2)))
    
    return euc_dist

prediction_dir='/Users/goutham/Documents/Senior_year/research_design/Test7PART1DLC_resnet50_Trial3Mar23shuffle1_10000.csv' 
rearing_dat='/Users/goutham/Documents/Senior_year/research_design/trial7_rearing_first10min.xlsx'

prediction_data=pd.read_csv(prediction_dir,header=1)
rearing_gt=pd.read_excel(rearing_dat,header=0)

pred_data_train=prediction_data.iloc[1:2175]  #this is the 5th minute to 7.5 minutes
pred_data_test=prediction_data.iloc[2176:4425]  #this is the 7.5th minute to the 10 minute

rearing_gt_train=rearing_gt.iloc[4576:6750,1] # add 5 second bias 5*15=75 b/c dow started 5 sec late
rearing_gt_test=rearing_gt.iloc[6751:,1]


snout_xy_train=pred_data_train[["snout","snout.1"]]
centroid_xy_train=pred_data_train[["centroid","centroid.1"]]
tailbase_xy_train=pred_data_train[["tailbase","tailbase.1"]]

snout_xy_test=pred_data_test[["snout","snout.1"]]
centroid_xy_test=pred_data_test[["centroid","centroid.1"]]
tailbase_xy_test=pred_data_test[["tailbase","tailbase.1"]]

snout_dat_train=snout_xy_train.to_numpy().astype(float)
centroid_dat_train=centroid_xy_train.to_numpy().astype(float)
tailbase_xy_train=tailbase_xy_train.to_numpy().astype(float)


snout_dat_test=snout_xy_test.to_numpy().astype(float)
centroid_dat_test=centroid_xy_test.to_numpy().astype(float)
tailbase_dat_test=tailbase_xy_test.to_numpy().astype(float)

rearing_gt_train=rearing_gt_train.to_numpy().astype(int)
rearing_gt_test=rearing_gt_test.to_numpy().astype(int)

snout_x_train=snout_dat_train[:,0]
snout_y_train=snout_dat_train[:,1]
centroid_x_train=centroid_dat_train[:,0]
centroid_y_train=centroid_dat_train[:,1]
tailbase_x_train=tailbase_xy_train[:,0]
tailbase_y_train=tailbase_xy_train[:,1]

snout_x_test=snout_dat_test[:,0]
snout_y_test=snout_dat_test[:,1]
centroid_x_test=centroid_dat_test[:,0]
centroid_y_test=centroid_dat_test[:,1]
tailbase_x_test=tailbase_dat_test[:,0]
tailbase_y_test=tailbase_dat_test[:,1]

distance_snout_centroid=Euclidean_distance(snout_x_train,snout_y_train,centroid_x_train,centroid_y_train)
distance_snout_tailbase=Euclidean_distance(snout_x_train,snout_y_train,tailbase_x_train,tailbase_y_train)

distance_snout_centroid_test=Euclidean_distance(snout_x_test,snout_y_test,centroid_x_test,centroid_y_test)
distance_snout_tail_test=Euclidean_distance(snout_x_test,snout_y_test,tailbase_x_test,tailbase_y_test)


#features=np.column_stack((snout_x_train,snout_y_train,centroid_x_train,centroid_y_train,tailbase_x_train,tailbase_y_train))
#features_test=np.column_stack((snout_x_test,snout_y_test,centroid_x_test,centroid_y_test,tailbase_x_test,tailbase_y_test))


# randomize columns for each festures+rearing_gt_train
# randomize columns for features_test+features_test


features=np.column_stack((distance_snout_centroid,distance_snout_tailbase))

features_test=np.column_stack((distance_snout_centroid_test,distance_snout_tail_test))

# try using balanced classes

clf = svm.SVC() # experimentally tried a bunch of Values for C and obtained best classificaiton on validation/testing of 
clf.fit(features,rearing_gt_train)

predictions=clf.predict(features_test)

loss=zero_one_loss(rearing_gt_test, predictions)
Accuracy=1-loss

print(Accuracy)

confusion=confusion_matrix(rearing_gt_test, predictions)
print(confusion)




