#-*-coding:utf-8-*-
from sklearn.model_selection import train_test_split
import os
import numpy as np
import PIL.Image as Image
path_train='/home/simple/mydemo/segmentation_models_mulclass/data/segment_wordlines/train/'
images_name=os.listdir(path_train+'img')


x_train, x_test = train_test_split(images_name,test_size=0.2, random_state=0)

path_val='/home/simple/mydemo/segmentation_models_mulclass/data/segment_wordlines/val/'
if len(os.listdir(path_val))==0:
        os.mkdir(os.path.join(path_val,'img'))
        os.mkdir(os.path.join(path_val,'mask'))


for index,label in enumerate(x_test):

    image=Image.open(os.path.join(path_train+'img',x_test[index]))
    image.save(os.path.join(path_val+'img',x_test[index]))
    os.remove(os.path.join(path_train+'img',x_test[index]))

    mask=Image.open(os.path.join(path_train+'mask',x_test[index]))
    mask.save(os.path.join(path_val+'mask',x_test[index]))
    os.remove(os.path.join(path_train+'mask',x_test[index]))