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
import time
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

    CLASSES = ['background', 'name', 'sex', 'minorities', 'year', 'month', 'day', 'location_1', 'location_2',
                       'location_3', 'id', 'issuing_authority_1', 'issuing_authority_2', 'validity_period']

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
        self.masks_fps = [os.path.join(masks_dir, image_id.split('.')[0] + '.png') for image_id in self.ids]

        # convert str names to class values on masks
        self.class_values = [self.CLASSES.index(cls.lower()) for cls in classes]

        self.augmentation = augmentation
        self.preprocessing = preprocessing

    def __getitem__(self, i):

        # read data
        image = cv2.imread(self.images_fps[i])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        mask = cv2.imread(self.masks_fps[i], 0)
        # mask[mask>0]=1
        # visualize(image=image,mask=mask*80)



        # extract certain classes from mask (e.g. cars)
        masks = [(mask == v) for v in self.class_values]
        mask = np.stack(masks, axis=-1)

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
        mask = np.argmax(mask, axis=0)
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

DATA_DIR = '/home/simple/mydemo/segmentation_models_mulclass/data/segment_wordlines/train/'

x_test_dir = os.path.join(DATA_DIR, 'img')
y_test_dir = os.path.join(DATA_DIR, 'mask')

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

# same image with different random transforms

ENCODER = 'resnet18'
ENCODER_WEIGHTS = 'imagenet'
CLASSES = ['background', 'name', 'sex', 'minorities', 'year', 'month', 'day', 'location_1', 'location_2',
                       'location_3', 'id', 'issuing_authority_1', 'issuing_authority_2', 'validity_period']
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
best_model = torch.load('/home/simple/mydemo/segmentation_models_mulclass/best_model.pth').cuda()
# create test dataset
test_dataset = Dataset(
    x_test_dir,
    y_test_dir,
    augmentation=get_validation_augmentation(),
    preprocessing=get_preprocessing(preprocessing_fn),
    classes=CLASSES,
)

test_dataloader = DataLoader(test_dataset)
# evaluate model on test set
# test_epoch = smp.utils.train.ValidEpoch(
#     model=best_model,
#     loss=loss,
#     metrics=metrics,
#     device=DEVICE,
# )
#
# logs = test_epoch.run(test_dataloader)
test_dataset_vis = Dataset(
    x_test_dir, y_test_dir,
    classes=CLASSES,
)
l=len(test_dataset)
for i in range(l):
    #n = np.random.choice(len(test_dataset))
    n=i
    image_vis = test_dataset_vis[n][0].astype('uint8')
    image, gt_mask = test_dataset[n]

    # gt_mask = gt_mask.squeeze().transpose((1,2,0))[:,:,9]
    gt_mask = gt_mask

    x_tensor = torch.from_numpy(image).to(DEVICE).unsqueeze(0)
    t1=time.clock()
    pr_mask = best_model.predict(x_tensor)
    t2=time.clock()
    #print(t2-t1)
    pr_mask = torch.argmax(pr_mask,dim=1).squeeze().cpu().numpy()
    #pr_mask = (pr_mask.squeeze().permute(1,2,0).cpu().numpy().round())[:,:,9]
    print(pr_mask.max())

    visualize(
        image=image_vis,
        ground_truth_mask=gt_mask,
        predicted_mask=pr_mask
    )
