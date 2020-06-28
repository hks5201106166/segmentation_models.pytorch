#-*-coding:utf-8-*-
#-*-coding:utf-8-*-
#-*-coding:utf-8-*-
#-*-coding:utf-8-*-

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import numpy as np
import cv2
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torch.utils.data import Dataset as BaseDataset
import torch
import numpy as np
import segmentation_models_pytorch as smp
import albumentations as albu
import time
import matplotlib.pyplot as plt
import torch.nn.functional as F
result_path='/home/simple/mydemo/ocr_project/segment/data/segment_idcard_result/'
if len(os.listdir(result_path))==0:
    os.mkdir(result_path+'drown_obverse')
    os.mkdir(result_path+'drown_reverse')
    os.mkdir(result_path+'up_obverse')
    os.mkdir(result_path+'up_reverse')
def crop_rect(img, rect,logo_mask):
    # get the parameter of the small rectangle
    boxs = cv2.boxPoints(rect)
    boxs = np.int0(boxs)
   # print(boxs)
    cv2.line(img, pt1=(boxs[0][0], boxs[0][1]), pt2=(boxs[1][0], boxs[1][1]), color=[255, 255, 255])
    cv2.line(img, pt1=(boxs[1][0], boxs[1][1]), pt2=(boxs[2][0], boxs[2][1]), color=[255, 255, 255])
    cv2.line(img, pt1=(boxs[2][0], boxs[2][1]), pt2=(boxs[3][0], boxs[3][1]), color=[255, 255, 255])
    cv2.line(img, pt1=(boxs[3][0], boxs[3][1]), pt2=(boxs[0][0], boxs[0][1]), color=[255, 255, 255])
    cv2.putText(img, str(rect[2]), (50, 50), cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 1.2, (255, 255, 255), 2)
    center, size, angle = (rect[0][0],rect[0][1]), (rect[1][0],rect[1][1]), rect[2]
    center, size = tuple(map(int, center)), tuple(map(int, size))
    # get row and col num in img
    width,height = rect[1][0],rect[1][1]
    if width < height:  # 计算角度，为后续做准备
        angle = 90+angle
        #print(angle)
        # calculate the rotation matrix
        M = cv2.getRotationMatrix2D(center, angle, 1)
        # rotate the original image
        img_rot = cv2.warpAffine(img, M,(img.shape[1],img.shape[0]))
        logo_mask_rot=cv2.warpAffine(logo_mask,M,(img.shape[1],img.shape[0]))
        # now rotated rectangle becomes vertical and we crop it
        img_crop = cv2.getRectSubPix(img_rot, (int(rect[1][1]),int(rect[1][0])), center)
        logo_mask_crop=cv2.getRectSubPix(logo_mask_rot,(int(rect[1][1]),int(rect[1][0])), center)
        height, width = img_crop.shape[0], img_crop.shape[1]
    else:
        angle = angle
       # print(angle)
        # calculate the rotation matrix
        M = cv2.getRotationMatrix2D(center, angle, 1)
        # rotate the original image
        img_rot = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]))
        logo_mask_rot = cv2.warpAffine(logo_mask, M, (img.shape[1], img.shape[0]))
        # now rotated rectangle becomes vertical and we crop it
        img_crop = cv2.getRectSubPix(img_rot, (int(rect[1][0]), int(rect[1][1])), center)
        logo_mask_crop = cv2.getRectSubPix(logo_mask_rot, (int(rect[1][0]), int(rect[1][1])), center)
        height, width = img_crop.shape[0], img_crop.shape[1]
    return img_crop,logo_mask_crop,img_rot
# helper function for data visualization
def visualize(**images):
    """PLot images in one row."""
    n = len(images)
    plt.figure(figsize=(16, 5))
    for i, (name, image) in enumerate(images.items()):
        plt.subplot(1, n, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.title(' '.join(name.split('_')).title())
        plt.imshow(image)
    plt.show()
def get_validation_augmentation():
    """Add paddings to make image shape divisible by 32"""
    test_transform = [
        albu.Resize(512, 512)
    ]
    return albu.Compose(test_transform)


def to_tensor(x, **kwargs):
    return x.transpose(2, 0, 1).astype('float32')


def get_preprocessing(preprocessing_fn):
    """Construct preprocessing transform

    Args:
        preprocessing_fn (callbale): data normalization function
            (can be specific for each pretrained neural network)
    Return:
        transform: albumentations.Compose

    """

    _transform = [
        albu.Lambda(image=preprocessing_fn),
        albu.Lambda(image=to_tensor),
    ]
    return albu.Compose(_transform)

# same image with different random transforms

ENCODER = 'resnet18'
ENCODER_WEIGHTS = 'imagenet'
CLASSES_idcard_detection = ['background','up_obverse','up_reverse','drown_obverse','drown_reverse']
CLASSES_logo_detection = ['background','logo']
ACTIVATION = 'softmax2d' # could be None for logits or 'softmax2d' for multicalss segmentation
DEVICE = 'cuda'

# create segmentation model with pretrained encoder
model = smp.Unet(
    encoder_name=ENCODER,
    encoder_weights=ENCODER_WEIGHTS,
    classes_idcard_detection=len(CLASSES_idcard_detection),
    classes_logo_detection=len(CLASSES_logo_detection),
    activation=ACTIVATION,
)

preprocessing_fn = smp.encoders.get_preprocessing_fn(ENCODER, ENCODER_WEIGHTS)




loss = smp.utils.losses.DiceLoss()
metrics = [
    smp.utils.metrics.IoU(threshold=0.5),
]

optimizer = torch.optim.Adam([
    dict(params=model.parameters(), lr=0.0001),
])





# load best saved checkpoint
best_model = torch.load('/home/simple/mydemo/ocr_project/segment/segmentation_models.pytorch/best_model.pth').cuda()
# create test dataset

path='/home/simple//mydemo/ocr_project/segment/test_data/'
train_or_test='test/'
# path='/home/simple/mydemo/segmentation_models_mulclass/'
#train_or_test='error_sample/'
images_name=os.listdir(path+train_or_test)
for i,image_name in enumerate(images_name):
    print(i)
    #n = np.random.choice(len(test_dataset))

    #image_vis = test_dataset_vis[n][0].astype('uint8')
    image = cv2.imread(path+train_or_test+image_name)
    #print(image_name)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    transform=get_validation_augmentation()
    image_resize=transform(image=image)['image']

    preprocessing=get_preprocessing(preprocessing_fn)
    image_cuda=preprocessing(image=image_resize)['image']


    #gt_mask = gt_mask.squeeze().transpose((1,2,0))[:,:,1]



    x_tensor = torch.from_numpy(image_cuda).to(DEVICE).unsqueeze(0)
    t1=time.clock()
    pr_mask,pr_mask_logo = best_model.predict(x_tensor)
    pr_mask=torch.argmax(torch.nn.Softmax2d()(pr_mask).squeeze(),dim=0).cpu().numpy()

    y_logo_detection = torch.nn.Softmax2d()(pr_mask_logo)
    logo_mask = y_logo_detection.cpu().numpy()
    logo_mask = np.uint8(np.argmax(logo_mask, axis=1)[0])
    logo_mask = cv2.resize(logo_mask,dsize=(1000,1000))
    # tt = np.sum(logo_mask)
    #cv2.imshow('hhjjh', logo_mask * 255)
    #cv2.waitKey(0)
    # ttt=pr_mask.squeeze().permute(1, 2, 0).cpu().numpy().round()[:,:,1]
    # cv2.imshow('hsk',np.uint8(pr_mask*50))
    # cv2.waitKey(10000)
    t2=time.clock()
    #print(t2-t1)
    nums_id = 0
    for index in range(1,5):

        mask =np.uint8(pr_mask==index)
        #print(np.sum(mask))
        if np.sum(mask)>10000:
            mask = cv2.resize(mask,dsize=(1000,1000))
            contours, hierarchy = cv2.findContours(mask.astype('uint8'), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)


            ls = []
            for contour in contours:
                l=cv2.contourArea(contour)
                ls.append(l)
            index_max=np.argmax(ls)
            rect=cv2.minAreaRect(contours[index_max])
            img_crop,logo_mask_crop,img_rot=crop_rect(image,rect,logo_mask)
            if index==1:
                os.mkdir(result_path+'/up_obverse/'+image_name.split('.jpg')[0])
                cv2.imwrite(result_path+'/up_obverse/'+image_name.split('.jpg')[0]+'/'+str(index)+image_name,img_crop)
                cv2.imwrite('/home/simple/mydemo/ocr_project/segment/data/data_image/' + str(i) + str(index)+image_name,
                            img_crop)
                cv2.imwrite(result_path+'/up_obverse/'+image_name.split('.jpg')[0]+'/'+'_mask_'+image_name,logo_mask_crop*255)
                cv2.imwrite('/home/simple/mydemo/ocr_project/segment/data/segmet_logo/train/mask/'+'_'+'0'+'_'+image_name,logo_mask_crop)
                cv2.imwrite('/home/simple/mydemo/ocr_project/segment/data/segmet_logo/train/img/'+'_'+'0'+'_'+image_name,img_crop)
                cv2.imwrite(result_path+'/up_obverse/' + image_name.split('.jpg')[0] + '/' + '_imagemask_' + image_name,cv2.addWeighted(img_crop[:,:,0],0.6,logo_mask_crop*255,0.4,0))
                nums_id+=1
            if index==2:
                img_crop = cv2.rotate(img_crop,rotateCode = cv2.ROTATE_180)
                logo_mask_crop=cv2.rotate(logo_mask_crop,rotateCode = cv2.ROTATE_180)
                os.mkdir(result_path+'/up_reverse/' + image_name.split('.jpg')[0])
                cv2.imwrite(result_path+'/up_reverse/' + image_name.split('.jpg')[0]+'/'+str(index)+image_name,img_crop)
                cv2.imwrite('/home/simple/mydemo/ocr_project/segment/data/data_image/' +str(i) + str(index) + image_name,
                            img_crop)
                cv2.imwrite('/home/simple/mydemo/ocr_project/segment/data/segmet_logo/train/mask/'+'_'+'0'+'_'+image_name,logo_mask_crop)
                cv2.imwrite('/home/simple/mydemo/ocr_project/segment/data/segmet_logo/train/img/'+'_'+'0'+'_'+image_name,img_crop)
                cv2.imwrite(result_path+'/up_reverse/' + image_name.split('.jpg')[0]+'/' + str(index)+'_mask_'+image_name,logo_mask_crop*255)
                cv2.imwrite(result_path+'/up_reverse/' + image_name.split('.jpg')[0] + '/' + str(index) + '_imagemask_' + image_name,cv2.addWeighted(img_crop[:,:,0],0.6,logo_mask_crop*255,0.4,0))
                nums_id += 1
            if index==3:
                os.mkdir(result_path+'/drown_obverse/' + image_name.split('.jpg')[0])
                cv2.imwrite(result_path+'/drown_obverse/' + image_name.split('.jpg')[0]+'/'+str(index)+image_name,img_crop)
                cv2.imwrite('/home/simple/mydemo/ocr_project/segment/data/data_image/' + str(i) + str(index) + image_name,
                            img_crop)
                cv2.imwrite('/home/simple/mydemo/ocr_project/segment/data/segmet_logo/train/mask/'+'_'+'1'+'_'+image_name,logo_mask_crop)
                cv2.imwrite('/home/simple/mydemo/ocr_project/segment/data/segmet_logo/train/img/'+'_'+'1'+'_'+image_name,img_crop)
                cv2.imwrite(result_path+'/drown_obverse/' + image_name.split('.jpg')[0] +'/'+ str(index)+'_imagemask_'+image_name,logo_mask_crop*255)
                cv2.imwrite(result_path+'/drown_obverse/' + image_name.split('.jpg')[0] + '/' + str(index) + '_mask_' + image_name, cv2.addWeighted(img_crop[:,:,0],0.6,logo_mask_crop*255,0.4,0))
                nums_id += 1
            if index==4:
                img_crop = cv2.rotate(img_crop, rotateCode=cv2.ROTATE_180)
                logo_mask_crop = cv2.rotate(logo_mask_crop, rotateCode=cv2.ROTATE_180)
                os.mkdir(result_path+'/drown_reverse/' + image_name.split('.jpg')[0])
                cv2.imwrite(result_path+'/drown_reverse/' + image_name.split('.jpg')[0]+'/'+str(index)+image_name,img_crop)
                cv2.imwrite('/home/simple/mydemo/ocr_project/segment/data/data_image/' + str(i) + str(index) + image_name,
                            img_crop)
                cv2.imwrite('/home/simple/mydemo/ocr_project/segment/data/segmet_logo/train/mask/'+'_'+'1'+'_'+image_name,logo_mask_crop)
                cv2.imwrite('/home/simple/mydemo/ocr_project/segment/data/segmet_logo/train/img/'+'_'+'1'+'_'+image_name,img_crop)
                cv2.imwrite(result_path+'/drown_reverse/' + image_name.split('.jpg')[0] +'/'+ str(index)+'_imagemask_'+image_name, logo_mask_crop*255+img_crop[:,:,0])
                cv2.imwrite(result_path+'/drown_reverse/' + image_name.split('.jpg')[0] + '/' + str(index) + '_mask_' + image_name,cv2.addWeighted(img_crop[:,:,0],0.6,logo_mask_crop*255,0.4,0))
                nums_id += 1
    if nums_id!=2:
        print('id segment is error')
        print(image_name)
        #break


    # visualize(
    #     image=image_vis,
    #     ground_truth_mask=gt_mask,
    #     predicted_mask=mask
    # )
