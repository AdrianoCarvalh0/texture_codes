import os
import torch
import numpy as np
from PIL import Image
import albumentations as aug
from functools import partial
from torchtrainer.imagedataset import ImageSegmentationDataset
from albumentations.pytorch import ToTensorV2
import scipy.ndimage as ndi

def zscore_orig(img, **kwargs):
    # return ((img-img.mean())/img.std()).astype(np.float32)
    # return img.astype(np.float32)
    return ((img-ORIG_MEAN)/ORIG_STD).astype(np.float32)

def zscore_art(img, **kwargs):
    return ((img-ART_MEAN)/ART_STD).astype(np.float32)

def name_to_label_map(img_path):
    return img_path.replace('.tiff', '.png')

def img_opener(img_path):
    img_pil = Image.open(img_path)
    return np.array(img_pil).astype(np.uint8)
    # return zscore(np.array(img_pil))[None]

def label_opener(img_path):
    pil_img = Image.open(img_path)
    if '.tiff' in str(img_path) or '.tif' in str(img_path):
        # if image is tif (0-255) pillow reads it as boolean
        # return torch.tensor(np.array(pil_img), dtype=torch.long)
        return np.array(pil_img, dtype=np.int64)
    else:
        return np.array(pil_img, dtype=np.int64)//255
        # return torch.tensor(np.array(pil_img), dtype=torch.long)//255

def luminosity(img, **kwargs):
    size_x, size_y = img.shape
    img_dist = np.ones((size_x, size_y))
    rand_x = np.random.randint(int(0.3*size_x), int(0.6*size_x))
    rand_y = np.random.randint(int(0.3*size_y), int(0.6*size_y))
    img_dist[rand_x, rand_y] = 0
    dt = ndi.distance_transform_edt(img_dist)
    dt = 1-(dt/np.max(dt))
    dt = dt**1.2

    return (img*dt).astype(np.float32)

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
            aug.GaussianBlur(blur_limit=(7, 11), p=1.),
            aug.OneOf([
                aug.RandomGamma(gamma_limit=(90, 250)),
                aug.RandomBrightnessContrast(brightness_limit=(-0.1, 0.3), 
                                             contrast_limit=(-0.2, 0.5)),
            ], p=1.),
            aug.GaussNoise(var_limit=(50, 250), p=1.),  # slow, 0.051s/img
            aug.Flip(),
            aug.Lambda(name='zscore', image=zscore_art, p=1.),
            aug.Lambda(name='luminosity', image=luminosity, p=1.),
            ToTensorV2(),
        ])
    elif type=='validation':
        transform_comp = aug.Compose([
            aug.Lambda(name='zscore', image=zscore_orig, p=1.),
            ToTensorV2()
        ])        

    transform_func = partial(transform, transform_comp=transform_comp)

    return transform_func

def dataset_stats(img_dir):
    dataset_pixels = []
    for fname in os.listdir(img_dir):
        img = np.asarray(Image.open(f'{img_dir}/{fname}'), dtype=np.float32)
        dataset_pixels += list(img.reshape(-1))
    dataset_pixels = np.array(dataset_pixels)

    return dataset_pixels.mean(), dataset_pixels.std()
    
def create_datasets(img_dir, label_dir, img_art_dir, label_art_dir, train_val_split, seed):
    global ORIG_MEAN, ORIG_STD, ART_MEAN, ART_STD
    ORIG_MEAN, ORIG_STD = dataset_stats(img_dir)
    ART_MEAN, ART_STD = dataset_stats(img_art_dir)

    name_to_label_map_art = lambda img_path: img_path
    
    ds_orig = ImageSegmentationDataset(img_dir, label_dir, name_to_label_map, img_opener=img_opener,
                                  label_opener=label_opener)
    _, ds_orig_valid = ds_orig.split_train_val(train_val_split, seed=seed)
    
    ds_art = ImageSegmentationDataset(img_art_dir, label_art_dir, name_to_label_map_art,
                                     img_opener=img_opener, label_opener=label_opener)
    ds_art_train, _ = ds_art.split_train_val(0., seed=seed)

    train_transform = create_transform(type='train-full')
    valid_transform = create_transform(type='validation')

    ds_art_train.set_transform(train_transform)
    ds_orig_valid.set_transform(valid_transform)
    
    return ds_art_train, ds_orig_valid