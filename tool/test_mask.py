#-*-coding:utf-8-*-
import cv2
import os
path='/home/simple/mydemo/ocr_project/segment/data/segment_idcard/val/new_mask/'
images=os.listdir(path)
for im in images:

    # image = cv2.imread(path+'imgs/'+im)
    # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)



   # mask_name=im.split('.')[0]+'.png'
    mask = cv2.imread(path+im)


   # cv2.imshow('image',image)
    cv2.imshow('mask',mask*80)
    cv2.waitKey(5000)