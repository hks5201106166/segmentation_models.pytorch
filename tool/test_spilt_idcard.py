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
if len(os.listdir('/home/simple/mydemo/segmentation_models_mulclass/test_results'))==0:
    os.mkdir('/home/simple/mydemo/segmentation_models_mulclass/test_results/drown_obverse')
    os.mkdir('/home/simple/mydemo/segmentation_models_mulclass/test_results/drown_reverse')
    os.mkdir('/home/simple/mydemo/segmentation_models_mulclass/test_results/up_obverse')
    os.mkdir('/home/simple/mydemo/segmentation_models_mulclass/test_results/up_reverse')
def crop_rect(img, rect):
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
        # now rotated rectangle becomes vertical and we crop it
        img_crop = cv2.getRectSubPix(img_rot, (int(rect[1][1]),int(rect[1][0])), center)
        height, width = img_crop.shape[0], img_crop.shape[1]
    else:
        angle = angle
       # print(angle)
        # calculate the rotation matrix
        M = cv2.getRotationMatrix2D(center, angle, 1)
        # rotate the original image
        img_rot = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]))
        # now rotated rectangle becomes vertical and we crop it
        img_crop = cv2.getRectSubPix(img_rot, (int(rect[1][0]), int(rect[1][1])), center)
        height, width = img_crop.shape[0], img_crop.shape[1]
    # if width < height:  # 计算角度，为后续做准备
    #     img_crop = np.rot90(img_crop)
    # plt.imshow(img_crop)
    # plt.show()
    # time.sleep(3)
    # cv2.imshow('img',img)
    # cv2.imshow('rot',img_rot)
    # cv2.imshow('crop',img_crop)
    #cv2.waitKey(1000)
    return img_crop, img_rot
class Dataset(BaseDataset):
    """CamVid Dataset. Read images, apply augmentation and preprocessing transformations.

    Args:
        images_dir (str): path to images folder
        masks_dir (str): path to segmentation masks folder
        class_values (list): values of classes to extract from segmentation mask
        augmentation (albumentations.Compose): data transfromation pipeline
            (e.g. flip, scale, etc.)
        preprocessing (albumentations.Compose): data preprocessing
            (e.g. noralization, shape manipulation, etc.)

    """

    CLASSES = ['background', 'id', 'id_reverse']

    def __init__(
            self,
            images_dir,
            masks_dir,
            classes=None,
            augmentation=None,
            preprocessing=None,
    ):
        self.ids = os.listdir(images_dir)
        self.images_fps = [os.path.join(images_dir, image_id) for image_id in self.ids]
        self.masks_fps = [os.path.join(masks_dir, image_id) for image_id in self.ids]

        # convert str names to class values on masks
        self.class_values = [self.CLASSES.index(cls.lower()) for cls in classes]

        self.augmentation = augmentation
        self.preprocessing = preprocessing

    def __getitem__(self, i):

        # read data
        image = cv2.imread(self.images_fps[i])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(self.masks_fps[i], 0)

        # extract certain classes from mask (e.g. cars)
        masks = [(mask == v) for v in self.class_values]
        mask = np.stack(masks, axis=-1).astype('float')

        # apply augmentations
        if self.augmentation:
            sample = self.augmentation(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']

        # apply preprocessing
        if self.preprocessing:
            sample = self.preprocessing(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']

        return image, mask

    def __len__(self):
        return len(self.ids)
# DATA_DIR = './data/CamVid/'
#
# # load repo with data if it is not exists
# if not os.path.exists(DATA_DIR):
#     print('Loading data...')
#     os.system('git clone https://github.com/alexgkendall/SegNet-Tutorial ./data')
#     print('Done!')




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
CLASSES = ['background','id','id_reverse']
ACTIVATION = 'softmax2d' # could be None for logits or 'softmax2d' for multicalss segmentation
DEVICE = 'cuda'

# create segmentation model with pretrained encoder
model = smp.FPN(
    encoder_name=ENCODER,
    encoder_weights=ENCODER_WEIGHTS,
    classes=len(CLASSES),
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
best_model = torch.load('/home/simple/mydemo/segmentation_models_mulclass/save_model/best_model.pth').cuda()
# create test dataset

path='/home/simple/mydemo/segmentation_models_mulclass/test_data/'
train_or_test='test/'
# path='/home/simple/mydemo/segmentation_models_mulclass/'
# train_or_test='error_data/'
images_name=os.listdir(path+train_or_test)
for image_name in images_name:
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
    pr_mask = best_model.predict(x_tensor)
    t2=time.clock()
    #print(t2-t1)
    nums_id = 0
    for index in range(1,5):
        mask = (pr_mask.squeeze().permute(1,2,0).cpu().numpy().round())[:,:,index]
        mask = cv2.resize(mask,dsize=(1000,1000))
        contours, hierarchy = cv2.findContours(mask.astype('uint8'), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        if len(contours)>0:
                ls = []
                for contour in contours:
                    l=len(contour)
                    ls.append(l)
                index_max=np.argmax(ls)
                if len(contours)>1:
                   print('111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111')
                    # visualize(
                    #     image=image,
                    #     predicted_mask=mask
                    # )

                if  len(contours[index_max])>30:
                    rect=cv2.minAreaRect(contours[index_max])
                    # nums=rect[1][0]+rect[1][1]
                    # if nums<5:
                    #     print(rect)
                    #     visualize(
                    #             image=image,
                    #             predicted_mask=mask
                    #         )
                    img_crop, img_rot=crop_rect(image,rect)
                    if index==1:
                        cv2.imwrite('/home/simple/mydemo/segmentation_models_mulclass/results_up_down/up/'+str(index)+image_name,img_crop)
                        nums_id+=1
                    if index==2:
                        img_crop = cv2.rotate(img_crop,rotateCode = cv2.ROTATE_180)
                        cv2.imwrite('/home/simple/mydemo/segmentation_models_mulclass/results_up_down/up/'+str(index)+image_name,img_crop)
                        nums_id += 1
                    if index==3:
                        cv2.imwrite('/home/simple/mydemo/segmentation_models_mulclass/results_up_down/down/'+str(index)+image_name,img_crop)
                        nums_id += 1
                    if index==4:
                        img_crop = cv2.rotate(img_crop, rotateCode=cv2.ROTATE_180)
                        cv2.imwrite('/home/simple/mydemo/segmentation_models_mulclass/results_up_down/down/'+str(index)+image_name,img_crop)
                        nums_id += 1
    if nums_id!=2:
        print(image_name)


    # visualize(
    #     image=image_vis,
    #     ground_truth_mask=gt_mask,
    #     predicted_mask=mask
    # )
