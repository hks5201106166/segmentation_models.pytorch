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
from torch.optim.lr_scheduler import MultiStepLR,StepLR
import PIL.Image as Image
import time
from imgaug.augmentables.segmaps import SegmentationMapsOnImage
from tool.Augmentation import get_validation_augmentation,get_training_augmentation



# backgourd:0
# up_obverse：1
# up_reverse：2
# drown_obverse：3
# drown_reverse：4
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

    #CLASSES = ['background','up_obverse','up_reverse','drown_obverse','drown_reverse']

    def __init__(
            self,
            images_dir,
            masks_dir,
            classes_idcard_detection=None,
            classes_logo_detection=None,
            augmentation=None,
            preprocessing=None,
    ):
        self.ids = os.listdir(images_dir)
        self.images_fps = [os.path.join(images_dir, image_id) for image_id in self.ids]
        self.masks_fps = [os.path.join(masks_dir, image_id.split('.')[0]) for image_id in self.ids]
        self.CLASSES_idcard_detection=classes_idcard_detection
        self.CLASSES_logo_detection=classes_logo_detection
        # convert str names to class values on masks
        self.class_values_idcard_detection = [self.CLASSES_idcard_detection.index(cls.lower()) for cls in classes_idcard_detection]
        self.class_values_logo_detection=[self.CLASSES_logo_detection.index(cls.lower()) for cls in classes_logo_detection]

        self.augmentation = augmentation
        self.preprocessing = preprocessing

    def __getitem__(self, i):

        # read data
        image = cv2.imread(self.images_fps[i])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        mask_file=self.masks_fps[i].split('/')[-1]
        mask_idcard_detection = cv2.imread(os.path.join(self.masks_fps[i],mask_file+'_1.png'), 0)
        mask_logo_detection=cv2.imread(os.path.join(self.masks_fps[i],mask_file+'_2.png'), 0)
        # img=image+mask_logo_detection*10
        #cv2.imshow('logo',mask_logo_detection*255)
        # cv2.imshow('ttt',img)
        # cv2.waitKey(3000000)
        #mask[mask>0]=1
        #visualize(image=image,mask=mask*80)



        # extract certain classes from mask (e.g. cars)
        mask_idcard_detection = [(mask_idcard_detection == v) for v in self.class_values_idcard_detection]
        #mask_idcard_detection = np.stack(mask_idcard_detection, axis=-1)

        mask_logo_detection = [(mask_logo_detection == v) for v in self.class_values_logo_detection]
        #mask_logo_detection = np.stack(mask_logo_detection, axis=-1)

        mask_idcard_detection.extend(mask_logo_detection)
        mask_idcard_logo_detection =mask_idcard_detection

        mask=np.stack(mask_idcard_logo_detection, axis=-1)
        # apply augmentations
        mask = SegmentationMapsOnImage(mask, shape=image.shape)

        # apply augmentations
        if self.augmentation:
            image, mask = self.augmentation(image=image, segmentation_maps=mask)
            mask = mask.arr
            # image, mask = sample['image'], sample['mask']

        # apply preprocessing
        if self.preprocessing:
            sample = self.preprocessing(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']

        mask_idcard_detection=mask[0:5,:,:]
        mask_logo_detection=mask[5:10,:,:]

        gt_mask_idcard_detection=mask_idcard_detection
        gt_mask_logo_detection = mask_logo_detection

        mask_idcard_detection=np.argmax(mask_idcard_detection,axis=0)
        mask_logo_detection=np.argmax(mask_logo_detection,axis=0)
        return image,(mask_idcard_detection,mask_logo_detection),(gt_mask_idcard_detection,gt_mask_logo_detection)

    def __len__(self):
        return len(self.ids)
DATA_DIR = '/home/simple/mydemo/ocr_project/segment/data/segment_idcard/'

x_train_dir = os.path.join(DATA_DIR, 'train/img')
y_train_dir = os.path.join(DATA_DIR, 'train/mask_with_logo')

x_valid_dir = os.path.join(DATA_DIR, 'val/img')
y_valid_dir = os.path.join(DATA_DIR, 'val/mask_with_logo')



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
        albu.Lambda(image=to_tensor, mask=to_tensor),
    ]
    return albu.Compose(_transform)
# augmented_dataset = Dataset(
#     x_train_dir,
#     y_train_dir,
#     augmentation=get_training_augmentation(),
#     classes=['background','id','id_reverse'],
# )

# same image with different random transforms
# for i in range(3):
#     image, mask = augmented_dataset[1]
#     visualize(image=image, mask=mask.squeeze(-1))

#ENCODER = 'se_resnext50_32x4d'
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
    activation=None,
)
#model = torch.load('/home/simple/mydemo/ocr_project/segment/segmentation_models.pytorch/best_model_6_16.pth').cuda()
preprocessing_fn = smp.encoders.get_preprocessing_fn(ENCODER, ENCODER_WEIGHTS)
train_dataset = Dataset(
    x_train_dir,
    y_train_dir,
    augmentation=get_training_augmentation(),
    preprocessing=get_preprocessing(preprocessing_fn),
    classes_idcard_detection=CLASSES_idcard_detection,
    classes_logo_detection=CLASSES_logo_detection
)

valid_dataset = Dataset(
    x_valid_dir,
    y_valid_dir,
    augmentation=get_validation_augmentation(),
    preprocessing=get_preprocessing(preprocessing_fn),
    classes_idcard_detection=CLASSES_idcard_detection,
    classes_logo_detection=CLASSES_logo_detection)

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=8)
valid_loader = DataLoader(valid_dataset, batch_size=8, shuffle=False, num_workers=8)
#loss=smp.utils.losses.DiceLoss(weight=torch.Tensor([0.5,1,1,1,1,1,1,1,1,1,1,1,1,1]))
#loss = smp.utils.losses.CrossEntropyLoss(weight=torch.Tensor([0.5,1,1,1,1,1,1,1,1,1,1,1,1,1]))
loss_idcard_detection= smp.utils.losses.CrossEntropyLoss()
loss_logo_detection=smp.utils.losses.CrossEntropyLoss()
#loss=torch.nn.CrossEntropyLoss()
metrics_idcard_detection = [
    smp.utils.metrics.IoU(threshold=0.5),
   # smp.utils.metrics.Accuracy(),
]
metrics_logo_detection = [
    smp.utils.metrics.IoU(threshold=0.5),
   # smp.utils.metrics.Accuracy(),
]
optimizer = torch.optim.Adam([
    dict(params=model.parameters(), lr=0.001),
])

schedular=MultiStepLR(optimizer=optimizer,milestones=[30,60])

# create epoch runners
# it is a simple loop of iterating over dataloader`s samples
train_epoch = smp.utils.train.TrainEpoch(
    model,
    loss_idcard_detection=loss_idcard_detection,
    loss_logo_detection=loss_logo_detection,
    metrics_idcard_detection=metrics_idcard_detection,
    metrics_logo_detection=metrics_logo_detection,
    optimizer=optimizer,
    device=DEVICE,
    verbose=True,
)

valid_epoch = smp.utils.train.ValidEpoch(
    model,
    loss_idcard_detection=loss_idcard_detection,
    loss_logo_detection=loss_logo_detection,
    metrics_idcard_detection=metrics_idcard_detection,
    metrics_logo_detection=metrics_logo_detection,
    device=DEVICE,
    verbose=True,
)

max_score = 0

for i in range(0, 100):

    print('\nEpoch: {},lr:{}'.format(i,optimizer.param_groups[0]['lr']))
    schedular.step(epoch=i)
    train_logs = train_epoch.run(train_loader)
    valid_logs = valid_epoch.run(valid_loader)

    torch.save(model, './best_model.pth')
    print('Model saved!')

    # do something (save model, change lr, etc.)
    # if max_score < valid_logs['iou_score']:
    #     max_score = valid_logs['iou_score']
    #     torch.save(model, './best_model.pth')
    #     print('Model saved!')

    # if i == 25:
    #     optimizer.param_groups[0]['lr'] = optimizer.param_groups[0]['lr']*0.1
    #     print('Decrease decoder learning rate to 1e-5!')
    # if i == 50:
    #     optimizer.param_groups[0]['lr'] = optimizer.param_groups[0]['lr']*0.1
    #     print('Decrease decoder learning rate to 1e-6!')
    # if i == 90:
    #     optimizer.param_groups[0]['lr'] = optimizer.param_groups[0]['lr']*0.1
    #     print('Decrease decoder learning rate to 1e-7!')


