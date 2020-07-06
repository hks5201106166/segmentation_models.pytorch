#-*-coding:utf-8-*-
import os
import cv2

path='/home/simple/mydemo/ocr_project/segment/data/test_results/'
images_name=os.listdir(os.path.join(path,'data_split'))
for index,image_name in enumerate(images_name):
    print(index)
    image=cv2.imread(os.path.join(path+'data_split',image_name),0)
    imageVar = cv2.Laplacian(image, cv2.CV_64F).var()
    if imageVar>300:
        cv2.imwrite(path+'data_split_clear/'+image_name,image)
    # print(imageVar)
    # cv2.imshow('img',image)
    # cv2.waitKey(1000)