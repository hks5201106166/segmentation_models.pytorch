#-*-coding:utf-8-*-
import os
import cv2
path='/home/simple/mydemo/ocr_project/segment/data/segment_idcard/train/new_mask/'
path_with_logo='/home/simple/mydemo/ocr_project/segment/data/segment_idcard/train/mask_with_logo/'
image_names=os.listdir(path)
for image_name in image_names:
    image=cv2.imread(path+image_name)
    dir_image_logo=image_name.split('.')[0]
    os.mkdir(os.path.join(path_with_logo,dir_image_logo))
    cv2.imwrite(os.path.join(path_with_logo+dir_image_logo,dir_image_logo+'_1.png'),image)
    cv2.imwrite(os.path.join(path_with_logo + dir_image_logo, dir_image_logo + '_2.png'), image)