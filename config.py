import cv2
from albumentations.augmentations.transforms import Equalize, Posterize, Downscale
from albumentations import (
    PadIfNeeded, HorizontalFlip, VerticalFlip, CenterCrop,    
    RandomCrop, Resize, Crop, Compose, HueSaturationValue,
    Transpose, RandomRotate90, ElasticTransform, GridDistortion, 
    OpticalDistortion, RandomSizedCrop, Resize, CenterCrop,
    VerticalFlip, HorizontalFlip, OneOf, CLAHE, Normalize,
    RandomBrightnessContrast, Cutout, RandomGamma, ShiftScaleRotate ,
    GaussNoise, Blur, MotionBlur, GaussianBlur, 
)

SEED = 69
n_epochs = 10
rate = 1.00
device = 'cuda:0'
data_dir = 'data/'
model_name= 'Hybrid' # Will come up with a better name later
img_model_name = 'resnet34'
nlp_model_name = 'bert-base-cased'
model_dir = 'model_dir'
history_dir = 'history_dir'
load_model = False
img_dim = 512
max_len = 64
batch_size = 32
learning_rate = 3e-3
mixed_precision = True
patience = 3
train_aug =Compose([
  ShiftScaleRotate(p=0.9,rotate_limit=180, border_mode= cv2.BORDER_REFLECT, value=[0, 0, 0], scale_limit=0.25),
    OneOf([
    Cutout(p=0.3, max_h_size=img_dim//16, max_w_size=img_dim//16, num_holes=10, fill_value=0),
    # GridMask(num_grid=7, p=0.7, fill_value=0)
    ], p=0.20),
    HueSaturationValue(p=0.4),
    Normalize(always_apply=True)
    ]
      )
val_aug = Compose([Normalize(always_apply=True)])
