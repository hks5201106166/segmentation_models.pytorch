#-*-coding:utf-8-*-
import xml.etree.ElementTree as ET
import os
import PIL.Image as Image
import json
import imageio
import numpy as np
from imgaug.augmentables.kps import KeypointsOnImage
from imgaug.augmentables.polys import Polygon
from imgaug.augmentables.bbs import BoundingBox
from imgaug.augmentables.segmaps import SegmentationMapsOnImage
import imgaug as ia
import matplotlib.pyplot as plot


# !/usr/bin/python
# -*- coding: UTF-8 -*-
# get annotation object bndbox location
import os
import cv2
import time
try:
    import xml.etree.cElementTree as ET  # 解析xml的c语言版的模块
except ImportError:
    import xml.etree.ElementTree as ET

#['background','name','sex','minorities','year','month','day','location_1','location_2','location_3','id','issuing_authority_1','issuing_authority_1','validity_period']
##get object annotation bndbox loc start
class_names=['background','name','sex','minorities','year','month','day','location_1','location_2','location_3','id','issuing_authority_1','issuing_authority_2','validity_period']
def GetAnnotBoxLoc(AnotPath):  # AnotPath VOC标注文件路径
    tree = ET.ElementTree(file=AnotPath)  # 打开文件，解析成一棵树型结构
    root = tree.getroot()  # 获取树型结构的根
    ObjectSet = root.findall('object')  # 找到文件中所有含有object关键字的地方，这些地方含有标注目标
    ObjBndBoxSet = {}  # 以目标类别为关键字，目标框为值组成的字典结构
    for Object in ObjectSet:
        ObjName = Object.find('name').text
        # if ObjName not in class_names:
        #     print('...................................................................................................')
        #     print(ObjName)
        BndBox = Object.find('bndbox')
        x1 = int(BndBox.find('xmin').text)  # -1 #-1是因为程序是按0作为起始位置的
        y1 = int(BndBox.find('ymin').text)  # -1
        x2 = int(BndBox.find('xmax').text)  # -1
        y2 = int(BndBox.find('ymax').text)  # -1
        BndBoxLoc = [x1, y1, x2, y2]
        if ObjName in ObjBndBoxSet:
            ObjBndBoxSet[ObjName].append(BndBoxLoc)  # 如果字典结构中含有这个类别了，那么这个目标框要追加到其值的末尾
        else:
            ObjBndBoxSet[ObjName] = [BndBoxLoc]  # 如果字典结构中没有这个类别，那么这个目标框就直接赋值给其值吧
    return ObjBndBoxSet


##get object annotation bndbox loc end

def display(objBox, pic):
    img = cv2.imread(pic)

    for key in objBox.keys():
        for i in range(len(objBox[key])):
            cv2.rectangle(img, (objBox[key][i][0], objBox[key][i][1]), (objBox[key][i][2], objBox[key][i][3]),
                          (0, 0, 255), 2)
            cv2.putText(img, key, (objBox[key][i][0], objBox[key][i][1]), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 1)
    cv2.imshow('img', img)
    cv2.imwrite('display.jpg', img)
    cv2.waitKey(3000)




path='/home/simple/mydemo/segmentation_models_mulclass/data/segment_wordlines/val/img'
images_name=os.listdir(path)
lens=[]
labels=[]
polygons=[]
for image_name in images_name:
    print('...........................................................................')
    xml_name=image_name.split('.')[0]+'.xml'
    image=imageio.imread('/home/simple/mydemo/segmentation_models_mulclass/data/segment_wordlines/val/img/'+image_name)
    #cv2.imshow('hks',image)
    image=image[:,:,0:3]
    #f = open('/home/simple/mydemo/segmentation_models_mulclass/data/json/'+js_name, encoding='utf-8')

    file = os.path.join('/home/simple/mydemo/segmentation_models_mulclass/data/segment_wordlines/xml/', xml_name)
    ObjBndBoxSet = GetAnnotBoxLoc(file)
    #print(ObjBndBoxSet)

    lens.append(len(ObjBndBoxSet))
    for item in ObjBndBoxSet.items():
        labels.append(item[0])
        rect=item[1][0]
        polygons.append(BoundingBox(rect[0],rect[1],rect[2],rect[3]).to_polygon())




    # labels=[shape['label'] for shape in shapes]
    # polygons=[Polygon(KeypointsOnImage(shape['points'],shape=image.shape).keypoints) for shape in shapes]
    segmap=np.zeros((image.shape[0],image.shape[1],14),dtype=np.uint8)
    #print(polygons)
    i=0
    for index,label in enumerate(labels):
        #backgourd:0
        #up_Obverse：1
        #up_reverse：2
        #drown_Obverse：3
        #droin_reverse：4
        #segmap = np.zeros((image.shape[0], image.shape[1], 3), dtype=np.uint8)
        class_names = ['background', 'name', 'sex', 'minorities', 'year', 'month', 'day', 'location_1', 'location_2',
                       'location_3', 'id', 'issuing_authority_1', 'issuing_authority_2', 'validity_period']
        color_temp=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

        if label == 'name':
            color_temp[1]=255
            segmap = polygons[index].draw_on_image(segmap,
                                                   color=color_temp,alpha=1.0, alpha_lines=0.0, alpha_points=0.0)
        if label == 'sex':
            color_temp[2] = 255
            segmap = polygons[index].draw_on_image(segmap,
                                                   color=color_temp, alpha=1.0, alpha_lines=0.0, alpha_points=0.0)

        if label == 'minorities':
            color_temp[3] = 255
            segmap = polygons[index].draw_on_image(segmap,
                                                   color=color_temp, alpha=1.0, alpha_lines=0.0, alpha_points=0.0)
        if label == 'year':
            color_temp[4] = 255
            segmap = polygons[index].draw_on_image(segmap,
                                                   color=color_temp, alpha=1.0, alpha_lines=0.0, alpha_points=0.0)
        if label == 'month':
            color_temp[5] = 255
            print(polygons[index])
            segmap = polygons[index].draw_on_image(segmap,
                                                   color=color_temp, alpha=1.0, alpha_lines=0.0, alpha_points=0.0)
        if label == 'day':
            color_temp[6] = 255
            segmap = polygons[index].draw_on_image(segmap,
                                                   color=color_temp, alpha=1.0, alpha_lines=0.0, alpha_points=0.0)
        if label == 'location_1':
            color_temp[7] = 255
            segmap = polygons[index].draw_on_image(segmap,
                                                   color=color_temp, alpha=1.0, alpha_lines=0.0, alpha_points=0.0)
        if label == 'location_2':
            color_temp[8] = 255
            segmap = polygons[index].draw_on_image(segmap,
                                                   color=color_temp, alpha=1.0, alpha_lines=0.0, alpha_points=0.0)
        if label == 'location_3':
            color_temp[9] = 255
            segmap = polygons[index].draw_on_image(segmap,
                                                   color=color_temp, alpha=1.0, alpha_lines=0.0, alpha_points=0.0)
        if label == 'id':
            color_temp[10] = 255
            segmap = polygons[index].draw_on_image(segmap,
                                                   color=color_temp, alpha=1.0, alpha_lines=0.0, alpha_points=0.0)
        if label == 'issuing_authority_1':
            color_temp[11] = 255
            segmap = polygons[index].draw_on_image(segmap,
                                                   color=color_temp, alpha=1.0, alpha_lines=0.0, alpha_points=0.0)
        if label == 'issuing_authority_2':
            color_temp[12] = 255
            segmap = polygons[index].draw_on_image(segmap,
                                                   color=color_temp, alpha=1.0, alpha_lines=0.0, alpha_points=0.0)
        if label == 'validity_period':
            color_temp[13] = 255
            segmap = polygons[index].draw_on_image(segmap,
                                                   color=color_temp, alpha=1.0, alpha_lines=0.0, alpha_points=0.0)



    data=segmap[:,:,5]
   # print(data.max())


    #@time.sleep(2)
    # ia.imshow(segmap[:, :, 1])
    # ia.imshow(segmap[:, :, 2])
    # ia.imshow(segmap[:, :, 3])
    # ia.imshow(segmap[:, :, 4])
    segmap=np.argmax(segmap,axis=2)
    # print(segmap.max())
    labels = []
    polygons = []
    # # print(segmap.max())
    segmap=segmap.astype(np.uint8)
    # cv2.imshow(str(np.random.randint(0, 199999)),segmap*50)
    # cv2.waitKey(3000000)
    imageio.imwrite('/home/simple/mydemo/segmentation_models_mulclass/data/segment_wordlines/val/mask/'+image_name.split('.')[0]+'.png',segmap)


    #ia.imshow(np.hstack([polygons[0].draw_on_image(image),polygons[1].draw_on_image(image)]))
    #segmap = SegmentationMapsOnImage(segmap, shape=image.shape)


    # visualize
    # Note that the segmentation map drawing methods return lists of RGB images.
    # That is because the segmentation map may have multiple channels
    # -- the C in (H,W,C) -- and one image is drawn for each of these channels.
    # We have C=1 here, so we get a list of a single image here and acces that via [0].
    #ia.imshow(segmap.draw_on_image(image)[0])


