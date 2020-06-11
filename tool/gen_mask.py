#-*-coding:utf-8-*-
import os
import PIL.Image as Image
import json
import imageio
import numpy as np
from imgaug.augmentables.kps import KeypointsOnImage
from imgaug.augmentables.polys import Polygon
from imgaug.augmentables.segmaps import SegmentationMapsOnImage
import imgaug as ia
import matplotlib.pyplot as plot
path='/home/simple/mydemo/segmentation_models_mulclass/data/train/img'
images_name=os.listdir(path)
for image_name in images_name:
    js_name=image_name.split('.')[0]+'.json'
    image=imageio.imread('/home/simple/mydemo/segmentation_models_mulclass/data/train/img/'+image_name)
    image=image[:,:,0:3]
    f = open('/home/simple/mydemo/segmentation_models_mulclass/data/json/'+js_name, encoding='utf-8')
    res = f.read()
    js=json.loads(res)
    shapes=js['shapes']
    labels=[shape['label'] for shape in shapes]
    polygons=[Polygon(KeypointsOnImage(shape['points'],shape=image.shape).keypoints) for shape in shapes]
    segmap=np.zeros((image.shape[0],image.shape[1],5),dtype=np.uint8)
    #print(polygons)
    for index,label in enumerate(labels):
        #backgourd:0
        #up_Obverse：1
        #up_reverse：2
        #drown_Obverse：3
        #droin_reverse：4
        #segmap = np.zeros((image.shape[0], image.shape[1], 3), dtype=np.uint8)
        if index==0 and label=='0':
            segmap = polygons[index].draw_on_image(segmap,
                                                   color=(0, 255, 0, 0, 0),
                                                   alpha=1.0, alpha_lines=0.0, alpha_points=0.0)
        if index==0 and label=='1':
            segmap = polygons[index].draw_on_image(segmap,
                                                   color=(0, 0, 255, 0, 0),
                                                   alpha=1.0, alpha_lines=0.0, alpha_points=0.0)
        if index == 1 and label == '0':
            segmap = polygons[index].draw_on_image(segmap,
                                                   color=(0, 0, 0, 255, 0),
                                                   alpha=1.0, alpha_lines=0.0, alpha_points=0.0)
        if index == 1 and label == '1':
            segmap = polygons[index].draw_on_image(segmap,
                                                   color=(0, 0, 0, 0, 255),
                                                   alpha=1.0, alpha_lines=0.0, alpha_points=0.0)

    # ia.imshow(segmap[:,:,0])
    # ia.imshow(segmap[:, :, 1])
    # ia.imshow(segmap[:, :, 2])
    # ia.imshow(segmap[:, :, 3])
    # ia.imshow(segmap[:, :, 4])
    # segmap=np.argmax(segmap,axis=2)
    # print(segmap.max())
    # segmap=segmap.astype(np.uint8)
    # imageio.imwrite('/home/simple/mydemo/segmentation_models_mulclass/data/val/new_mask/'+image_name,segmap)


    #ia.imshow(np.hstack([polygons[0].draw_on_image(image),polygons[1].draw_on_image(image)]))
    #segmap = SegmentationMapsOnImage(segmap, shape=image.shape)


    # visualize
    # Note that the segmentation map drawing methods return lists of RGB images.
    # That is because the segmentation map may have multiple channels
    # -- the C in (H,W,C) -- and one image is drawn for each of these channels.
    # We have C=1 here, so we get a list of a single image here and acces that via [0].
    #ia.imshow(segmap.draw_on_image(image)[0])


