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
best_model = torch.load('/home/simple/mydemo/ocr_project/segment/segmentation_models.pytorch/best_model.pth').cuda()
# create test dataset

path='/home/simple//mydemo/ocr_project/segment/data/segmet_logo/remove_logo_and_aug_image3/train'
train_or_test='/'
# path='/home/simple/mydemo/segmentation_models_mulclass/'
# train_or_test='error_data/'
# images_name=os.listdir(path+train_or_test)
images_name=os.listdir('/home/simple/mydemo/ocr_project/segment/data/segmet_logo/remove_logo_and_aug_image3/train/')
for index,image_name in enumerate(images_name):
    print(index)
    #n = np.random.choice(len(test_dataset))

    #image_vis = test_dataset_vis[n][0].astype('uint8')
    image = cv2.imread('/home/simple/mydemo/ocr_project/segment/data/segmet_logo/remove_logo_and_aug_image3/train/'+image_name)
    #print(image_name)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image1 = image[:, 449:898, :]
    image = image[:, 0:449, :]


    # cv2.imshow('mask', image)
    # cv2.waitKey(0)
    transform=get_validation_augmentation()
    image_resize=transform(image=image)['image']

    preprocessing=get_preprocessing(preprocessing_fn)
    image_cuda=preprocessing(image=image_resize)['image']


    #gt_mask = gt_mask.squeeze().transpose((1,2,0))[:,:,1]



    x_tensor = torch.from_numpy(image_cuda).to(DEVICE).unsqueeze(0)
    t1=time.clock()
    pr_mask = best_model.predict(x_tensor)
    y_logo_detection = torch.nn.Softmax2d()(pr_mask)
    logo_mask = y_logo_detection.cpu().numpy()
    logo_mask = np.uint8(np.argmax(logo_mask, axis=1)[0]*255)
    logo_mask = cv2.resize(logo_mask, dsize=(image.shape[1],image.shape[0]))
    kernel = np.ones((10, 10), np.uint8)
    logo_mask = cv2.erode(logo_mask, kernel=np.ones((5, 5), np.uint8))
    logo_mask = cv2.dilate(logo_mask, kernel=np.ones((10, 10), np.uint8))
    contours, hierarchy = cv2.findContours(logo_mask.astype('uint8'), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    ls = []
    for contour in contours:
        l = cv2.contourArea(contour)
        ls.append(l)
    index_max = np.argmax(ls)
    x, y, w, h = cv2.boundingRect(contours[index_max])
    # cv2.rectangle(image, (x, y), (x + w, y + h), (255, 255, 255), 2)
    image_roi=image[y:y+h,x:x+w]
    image1_roi=image1[y:y+h,x:x+w]
    train_image=np.hstack([image1_roi,image_roi])
    cv2.imwrite('/home/simple/mydemo/ocr_project/segment/data/segmet_logo/data_remove_the_logo/'+image_name,train_image)
    # cv2.imshow('mask',train_image)
    # cv2.waitKey(1000)
    t2=time.clock()
    #print(t2-t1)



    # visualize(
    #     image=image_vis,
    #     ground_truth_mask=gt_mask,
    #     predicted_mask=mask
    # )
