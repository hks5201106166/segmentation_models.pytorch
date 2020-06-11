#-*-coding:utf-8-*-
import cv2
import os
path='/home/simple/segmentation_models.pytorch-master/data/'
images=os.listdir(path+'imgs')
for im in images:

    image = cv2.imread(path+'imgs/'+im)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)



    mask_name=im.split('.')[0]+'.png'
    mask = cv2.imread(path+'cv_mask/'+mask_name)


    cv2.imshow('image',image)
    cv2.imshow('mask',mask*80)
    cv2.waitKey(5000)