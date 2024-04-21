import torch
import numpy as np
from PIL import Image
from torchtrainer.imagedataset import ImageSegmentationDataset
import os
import albumentations as aug
from functools import partial
from torchtrainer.imagedataset import ImageSegmentationDataset
from albumentations.pytorch import ToTensorV2
import scipy.ndimage as ndi

def dataset_stats(img_dir):
    dataset_pixels = []
    for fname in os.listdir(img_dir):
        img = np.asarray(Image.open(f'{img_dir}/{fname}'), dtype=np.float32)
        dataset_pixels += list(img.reshape(-1))
    dataset_pixels = np.array(dataset_pixels)

    return dataset_pixels.mean(), dataset_pixels.std()

def name_to_label_map(img_path):
    return img_path.replace('.tiff', '.png')

def zscore(img, **kwargs):
    return ((img-ORIG_MEAN)/ORIG_STD).astype(np.float32)

def transform(image, mask, transform_comp):
    """Given an image and a mask, apply transforms in `transform_comp`. This function is useful 
    because albumentations transforms need the image= and mask= keywords as input, and it also 
    returns a dictionary. Using this function, we can call the transforms as 
    
    image_t, label_t = transform(image, label)
    
    instead of having to deal with dictionaries."""

    res = transform_comp(image=image, mask=mask)
    image, mask = res['image'], res['mask']

    return image, mask.long()

def create_transform(type='none'):
    """Create a transform function with signature transform(image, label)."""

    if type=='none':
        transform_comp = aug.Compose([ToTensorV2()])
    if type=='train-full':
        transform_comp = aug.Compose([
            aug.Flip(),
            aug.Lambda(name='zscore', image=zscore, p=1.),
            ToTensorV2(),
        ])
    elif type=='validation':
        transform_comp = aug.Compose([
            aug.Lambda(name='zscore', image=zscore, p=1.),
            ToTensorV2()
        ])

    transform_func = partial(transform, transform_comp=transform_comp)

    return transform_func

def img_opener(img_path):
    img_pil = Image.open(img_path)
    return np.array(img_pil).astype(np.uint8)
    # return zscore(np.array(img_pil))[None]

def label_opener(img_path):
    pil_img = Image.open(img_path)
    return np.array(pil_img).astype(np.uint8)//255
    # return torch.tensor(np.array(pil_img), dtype=torch.long)//255

def create_datasets(img_dir, label_dir, train_val_split, seed):
    global ORIG_MEAN, ORIG_STD 
    ORIG_MEAN, ORIG_STD =  dataset_stats(img_dir)
    ds = ImageSegmentationDataset(img_dir, label_dir, name_to_label_map, img_opener=img_opener,
                                  label_opener=label_opener)
    ds_train, ds_valid = ds.split_train_val(train_val_split, seed=seed)

    train_transform = create_transform(type='train-full')
    valid_transform = create_transform(type='validation')

    # ds_train.set_transform(train_transform)
    ds_train.set_transform(valid_transform) # without data augmentation
    ds_valid.set_transform(valid_transform)
    
    return ds_train, ds_valid