#!/usr/bin/env python
# coding: utf-8

# The environment setup is copied from [Pednoi](https://www.kaggle.com/pednoi), and roughly follows the setup prescribed [here.](https://github.com/matterport/Mask_RCNN/blob/master/samples/shapes/train_shapes.ipynb>) What follows after is a thorough analysis of Mask R-CNN trained on the iMaterialist dataset.

# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from pathlib import Path
import os
import sys
import argparse
# !pip install tensorflow==2.0.0-beta1
import tensorflow as tf
import json
import multiprocessing

from tensorflow.python.client import device_lib

def get_available_gpus():
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos]
get_available_gpus()
print("Core count: {}".format(multiprocessing.cpu_count()))

# INPUT_DIR = Path('/kaggle/input')
# DATA_DIR = INPUT_DIR/"imaterialist-fashion-2019-FGVC6"
# ROOT_DIR = Path('/kaggle/working')
# INPUT_DIR = Path('/kaggle/input')
# # get_ipython().system('pwd')
DATA_DIR = Path('/home/chadwick/Documents/ImageSegmentation/iMaterialistFashion2019/imaterialist-fashion-2019-FGVC6')
ROOT_DIR = Path('/home/chadwick/Documents/ImageSegmentation/iMaterialistFashion2019')


# Actually, unfamiliar with how Kaggle handles their filesystem. Let's check.

# print(os.listdir("/kaggle/"))
# print(os.listdir("/kaggle/config"))
# print(os.listdir(INPUT_DIR))
# print(os.listdir(ROOT_DIR))
# print(os.listdir(DATA_DIR))


# ### Get the Matterport implementation of Mask R-CNN from github

# # get_ipython().system('git clone https://www.github.com/matterport/Mask_RCNN.git')


# print("Current Directory: {}".format(os.curdir))
# print("Current Directory Contents: {}".format(os.listdir()))
# print("Changing dir...")
# print(os.listdir('.'))
# os.chdir('Mask_RCNN')
# print("Current Directory: {}".format(os.curdir))
# print("Current Directory Contents: {}".format(os.listdir()))

# get_ipython().system('rm -rf .git # to prevent an error when the kernel is committed')
# get_ipython().system('rm -rf images assets # to prevent displaying images at the bottom of a kernel')


### Matterport Imports ###
import os
import sys
import random
import math
import re
import time
import numpy as np
import cv2
import matplotlib
import matplotlib.pyplot as plt
import PIL
from PIL import Image, ImageFile

from imgaug import augmenters as iaa
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.preprocessing import MultiLabelBinarizer
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold, MultilabelStratifiedShuffleSplit
from scipy.ndimage import zoom

# Import Mask RCNN
sys.path.append(str(ROOT_DIR/'Mask_RCNN'))
print(sys.path)
from mrcnn.config import Config
from mrcnn import utils
import mrcnn.model as modellib
from mrcnn import visualize
from mrcnn.model import log


# get_ipython().system('wget --quiet https://github.com/matterport/Mask_RCNN/releases/download/v2.0/mask_rcnn_coco.h5')
# print(os.listdir())
# get_ipython().system('ls -lh mask_rcnn_coco.h5')

ImageFile.LOAD_TRUNCATED_IMAGES = True
MAX_DIM = 512
# COCO_WEIGHTS_PATH = 'mask_rcnn_coco.h5'
# # WEIGHTS_PATH = COCO_WEIGHTS_PATH

# ### The Matterport library has two primary structures to work with: Config & Dataset

# Config first...

class iMaterialistConfig(Config):
    def __init__(self):
        super(iMaterialistConfig, self).__init__()

        """Configuration for training on the iMaterialist Fashion 2019 dataset.
        Derives from the base Config class and overrides values specific to the 
        iMaterialist dataset."""
        
        # Give config a name
        self.NAME = "fashion"
        
        # Train with 1 img/GPU, as the images are large, and I haven't done the
        # calculation to determine if we can fit the largest image in memory. 
        # Trial by fire!
        self.GPU_COUNT = 1
        self.IMAGES_PER_GPU = 2
        
        # At first, we will ignore attributes to simplify the classification
        self.NUM_CLASSES = 46 + 1 # background + 27 main articles + 19 apparel parts
        
        # We will try to train on the images as is. This may fail, as the
        # largest image in the set is (6824, 10717). Yet, we push on. 
        self.IMAGE_MIN_DIM = 151
        self.IMAGE_MAX_DIM = 16384
    #     IMAGE_RESIZE_MODE = 'none'
        
        # The objects in the images were analyzed in my other notebook on this
        # challenge. Good sizes to start seemed to be 128, 384, and 896.
        self.RPN_ANCHOR_SCALES = (128, 384, 896)
        
        # Apparently the Mask RCNN paper prescribes 512, but this is alot, especially
        # for this dataset. We trim this down to a more reasonable number. 
        # TRAIN_ROIS_PER_IMAGE = 45
        self.TRAIN_ROIS_PER_IMAGE = 400

        # Use a small step size at first (don't need validation updates very often)
        # Ideally this uses all 0of the instances in the training set. So this should 
        # be equal to (len(training_dataset)/(GPU_COUNT*IMAGES_PER_GPU)). Validation
        # steps should similarly mimic the above calc.
        # STEPS_PER_EPOCH = 1000        # Batches agnostic
        # STEPS_PER_EPOCH = 9125        # Batches = 4
        # STEPS_PER_EPOCH = 7300        # Batches = 5
        # STEPS_PER_EPOCH = 6083        # Batches = 6
        # All of these values for steps require lots of computation time. Let's
        # instead train half the data in each epoch.
        # STEPS_PER_EPOCH = 3042        # Batches = 6, 2 epochs for full set
        # STEPS_PER_EPOCH = 3650        # Batches = 5, 2 epochs for full set
        self.STEPS_PER_EPOCH = 4563        # Batches = 5, 2 epochs for full set

        # Validation steps to run after each epoch
        # VALIDATION_STEPS = 200        # Batches agnostic
        # VALIDATION_STEPS = 2281       # Batches = 4
        # VALIDATION_STEPS = 1825       # Batches = 5
        # VALIDATION_STEPS = 1520       # Batches = 6
        # All of these values for steps require lots of computation time. Let's
        # instead train half the data in each epoch.
        # VALIDATION_STEPS = 760        # Batches = 6, 2 epochs for full set
        # VALIDATION_STEPS = 913        # Batches = 5, 2 epochs for full set
        self.VALIDATION_STEPS = 1141        # Batches = 5, 2 epochs for full set
        
        self.BACKBONE = "resnet50"

        self.compute_attributes()

# config = iMaterialistConfig()
# config.display()



# Dataset second...  
#   
# We need to override the following methods:
# * load_image
# * load_mask
# * image_reference

import json
from pandas.io.json import json_normalize

# print("Train df grouped by image:")
# print(type(train_df_img_groups))
# print(train_df_img_groups.head())
# print()
# print("First group of train df grouped by image:")
# df_iterable = list(zip(*train_df_img_groups.iterrows()))
# img_ids, row_list = df_iterable[0], df_iterable[1]
# print(row_list[0])


class iMaterialistDataset(utils.Dataset):
    """Generates the iMaterialist Fashion 2019 dataset"""
    
    def __init__(self, df, category_names):
        super().__init__(self)
        self.category_names = category_names
        
        ##########################################################
        # add_class() fills in self.class_info, a list of dicts,
        # so we can keep track of labels in training and test
        #
        # The entries in self.class_info list take the form:
        # {
        #     "source": source,
        #     "id": class_id,
        #     "name": class_name,
        # }
        #
        for i, category_name in enumerate(category_names):
            # i+1 because background is id = 0
            self.add_class(source = "fashion", 
                           class_id = i+1, 
                           class_name = category_name
                          )
        
        ##########################################################
        # add_image() fills in self.image_info, a list of 
        # dicts, so we can generate appropriate masks
        #
        # The entries in self.image_info template looks like this:
        # {
        #     "id": image_id,
        #     "source": source,
        #     "path": path,
        # }
        #
        # You can add custom fields to keep track of things like
        # mask pixel encodings, etc.
        #
        for img_id, row in df.iterrows():
            self.add_image(source = "fashion",
                           image_id = row.name,
                           path = DATA_DIR/("train/"+row.name),
                           labels = row["CategoryIds"],
                           width = row["Width"],
                           height = row["Height"],
                           annotations = row["EncodedPixels"]
                          )
        
    def load_image(self, img_num):
        """Load the specified image and return a [H,W,3] Numpy array.
        """
        # Given image_name (<string>.jpg), load the image into array
        info = self.image_info[img_num]
        img = np.asarray(Image.open(info["path"]))
        return img
    
    def RLE_to_submask(self, mask, encoded_pixels):
        """Fills in 1's in mask array for segmented object that is
        encoded with values in encoded pixels.
        """
        mask_height = mask.shape[0]
        start_pixels = encoded_pixels[::2]
        lengths = encoded_pixels[1:][::2]
        for start_pixel, length in zip(start_pixels, lengths):
            start_pixel, length = int(start_pixel), int(length)
            # Pixels are numbered in y (not left to right in x)
            y_start = (start_pixel-1) % mask_height
            x_start = int((start_pixel - 1) / mask_height)
            mask[y_start: y_start + length, x_start] = 1
        return mask

    def load_mask(self, img_num):
        """Load instance masks for the given image.
        Different datasets use different ways to store masks. Override this
        method to load instance masks and return them in the form of an
        array of binary masks of shape [height, width, instances].
        
        Returns:
            masks: A bool array of shape [height, width, instance count] with
                a binary mask per instance.
            class_ids: a 1D array of class IDs of the instance masks.
        """
        info = self.image_info[img_num]
        masks = np.full((info["height"], info["width"], len(info["labels"])), fill_value=0, dtype=np.uint8)
        category_ids = []
        for instance_num, category_annotation in enumerate(zip(info["labels"], info["annotations"])):
            category_id, encoded_pixels = category_annotation
            encoded_pixels = encoded_pixels.split(' ')
            mask = masks[:, :, instance_num]
            mask = self.RLE_to_submask(mask, encoded_pixels)
            masks[:, :, instance_num] = mask
            category_ids.append(int(category_id)+1) # Offset by 1 for BG at index 0.
        category_ids = np.array(category_ids, dtype=np.int32)
        return masks,category_ids

    def image_reference(self, img_num):
        info = self.image_info[img_num]
        if info["source"] == "fashion":
            # Match the category_id (x for x in info.labels) with the string in
            # the category names list.
            return info["path"], [self.category_names[int(x)] for x in info["labels"]]
        else:
            super(self.__class__).image_reference(self, image_id)

# dataset = iMaterialistDataset(train_df_img_groups, category_names)
# dataset.prepare()

def sample_dataset(dataset):
    for i in range(6):
        image_id = random.choice(dataset.image_ids)
        print("ImageId: {}\n".format(image_id))
        print(dataset.image_reference(image_id))

        image = dataset.load_image(image_id)
        mask, class_ids = dataset.load_mask(image_id)
        visualize.display_top_masks(image, mask, class_ids, dataset.class_names, limit=4)
        
# sample_dataset(dataset)

# # Let's train this muthasucka!
# ![alt text](https://thenypost.files.wordpress.com/2014/10/2220943-e1413627640425.jpg "George Clinton of Parliament! Check out that fashion though.")

# This code partially supports k-fold training, 
# you can specify the fold to train and the total number of folds here
# FOLD = 0
# N_FOLDS = 5

# kf = KFold(n_splits=N_FOLDS, random_state=42, shuffle=True)
# splits = kf.split(train_df_img_groups) # ideally, this should be multilabel stratification

# def get_fold():    
#     for i, (train_index, valid_index) in enumerate(splits):
#         if i == FOLD:
#             return train_df_img_groups.iloc[train_index], train_df_img_groups.iloc[valid_index]
        
# train_fold, valid_fold = get_fold()

# train_dataset = iMaterialistDataset(train_fold, category_names)
# train_dataset.prepare()

# valid_dataset = iMaterialistDataset(valid_fold, category_names)
# valid_dataset.prepare()


# Initial go. These are the parameters that Pednoi uses.
# LR = 2e-3
# EPOCHS = [2, 6, 8]

# import warnings
# warnings.filterwarnings("ignore")


# EDIT: This won't work with current config

# model = modellib.MaskRCNN(mode='training', config=config, model_dir=ROOT_DIR)

# model.load_weights(COCO_WEIGHTS_PATH, 
#                    by_name=True, 
#                    exclude=['mrcnn_class_logits', 
#                             'mrcnn_bbox_fc', 
#                             'mrcnn_bbox', 
#                             'mrcnn_mask'])


# First we train the heads, which is a type of transfer learning known as <u>Feature Extraction</u>. This trains the classification pieces of the network and freezes any and all convolutional layers. The idea is that the network uses features already learned by the backbone network (in this case ResNet50), which are likely useful to learn features about various articles of clothing/fashion.

# Below doesn't seem to work (kernel dies). Let's try to resize images as Pednoi does.

# Train the head branches
# Passing layers="heads" freezes all layers except the head
# layers. You can also pass a regular expression to select
# which layers to train by name pattern.


# model.train(train_dataset, valid_dataset, 
#             learning_rate=LR, 
#             epochs=EPOCHS[0], 
#             layers='heads')

# history = model.keras_model.history.history


# We will implement a resize for images that <u>need</u> it. Let's do Pednoi's resize to a max image dimension of 512 for now. That means we will rescale images that have max dimension > 512 so that their max dimension is 512. Lose some detail here, probably OK.  
#   
# This will include edits to the following objects/methods:
# * iMaterialistConfig() - IMAGE_MAX_DIM, RPN_ANCHOR_SCALES, IMAGES_PER_GPU
# * iMaterialistDataset() - load_image(), load_mask()

# Tiny image resize experiment to determine which is most appropriate
# Seek out the largest images in the data set and resize to where the max dimension is 512
# interp_methods = [PIL.Image.NEAREST, 
#                   PIL.Image.BILINEAR, 
#                   PIL.Image.BICUBIC, 
#                   PIL.Image.LANCZOS, 
#                  ]
# MAX_DIM = 512

# max_width_img_id = train_df_img_groups["Width"].idxmax()
# max_height_img_id = train_df_img_groups["Height"].idxmax()

def resize_interp(img_id):
    print(max_width_img_id)
    img_info = train_df_img_groups.ix[img_id]
    img = Image.open(DATA_DIR/("train/"+img_id))

    fig, ax = plt.subplots(len(interp_methods), 1, figsize=(16, 30))
    for i, interp_method in enumerate(interp_methods):
        width, height = img_info["Width"], img_info["Height"]
        ratio = min(MAX_DIM/width, MAX_DIM/height)
        img_resize = img.resize((int(ratio*width), int(ratio*height)), 
                                                    resample=interp_method)
        ax[i].imshow(img_resize)

# resize_interp(max_width_img_id)
# resize_interp(max_height_img_id)


# Honestly, Lanczos looks like the best representation. Let's use it!

# Mask scaling is a slightly different beast. In `load_image()`, masks are arrays and not PIL Images. Let's see what zoom can do for us:


# fig, ax = plt.subplots(2, 4, figsize=(16,16))
# image_id = random.choice(dataset.image_ids)
# print("ImageId: {}\n".format(image_id))
# print(dataset.image_reference(image_id))
# image = dataset.load_image(image_id)
# mask, class_ids = dataset.load_mask(image_id)
# mask = mask[:, :, 0]
# print(mask.shape)
# width, height = mask.shape

# Resize so the max dimension for each image is MAX_DIM in size
# ratio = min(MAX_DIM/width, MAX_DIM/height)
# resize_shape = (int(ratio*width), int(ratio*height))

# ax[0, 0].imshow(mask)
# zoom_mask = zoom(mask, ratio, order=1, mode='nearest')
# ax[0, 1].imshow(zoom_mask)
# zoom_mask = zoom(mask, ratio, order=3, mode='nearest')
# ax[0, 2].imshow(zoom_mask)
# zoom_mask = zoom(mask, ratio, order=5, mode='nearest')
# ax[0, 3].imshow(zoom_mask)

# ax[1, 0].imshow(mask)
# zoom_mask = zoom(mask, ratio, order=3, mode='reflect')
# ax[1, 1].imshow(zoom_mask)
# zoom_mask = zoom(mask, ratio, order=3, mode='nearest')
# ax[1, 2].imshow(zoom_mask)
# zoom_mask = zoom(mask, ratio, order=3, mode='wrap')
# ax[1, 3].imshow(zoom_mask)


# There are weird slim margins appearing for masks that have objects touching the border. 
# Looks like there's not much reason to pick hairs here though, so let's go with default settings

class iMaterialistMaxDimConfig(iMaterialistConfig):
    def __init__(self):
        super(iMaterialistMaxDimConfig, self).__init__()
        """Configuration for training on the iMaterialist Fashion 2019 dataset.
        Derives from the base Config class and overrides values specific to the 
        iMaterialist dataset. In addition"""
        
        # Give config a name
        self.NAME = "fashion"
        
        # Train with 4 imgs/GPU, as the images are large as Pednoi prescribes.
        self.GPU_COUNT = 1
        # self.IMAGES_PER_GPU = 8
        # self.IMAGES_PER_GPU = 5
        # self.IMAGES_PER_GPU = 4
        self.IMAGES_PER_GPU = 3
        # self.IMAGES_PER_GPU = 2
        
        # We will try to train on resized image data.
        self.IMAGE_MIN_DIM = MAX_DIM
        self.IMAGE_MAX_DIM = MAX_DIM
        
        # The objects in the images were analyzed in my other notebook on this
        # challenge. Good sizes to start seemed to be 128, 384, and 896 for 
        # non-resized images. Let's try the following:
        self.RPN_ANCHOR_SCALES = (32, 64, 128, 256, 384)

        # Maximum number of ground truth instances to use in one image
        # Since this effects memory usage, make it smaller
        self.MAX_GT_INSTANCES = 200
        # self.MAX_GT_INSTANCES = 100

        # Gradient Clip Norm
        # As prescribed here https://medium.com/@connorandtrent/
        # mask-r-cnn-hyperparameter-experiments-with-weights-and-biases-bd2319faae26
        self.GRADIENT_CLIP_NORM = 10.0

        # As prescribed here https://medium.com/@connorandtrent/
        # mask-r-cnn-hyperparameter-experiments-with-weights-and-biases-bd2319faae26
        # self.BACKBONE = "resnet50"
        self.BACKBONE = "resnet101"

        # Weight Decay (Regularization)
        self.WEIGHT_DECAY = 1e-4
        # self.WEIGHT_DECAY = 1e-5

        # self.LEARNING_MOMENTUM = 0.95

        self.compute_attributes()


# config = iMaterialistMaxDimConfig()
# config.display()



class iMaterialistMaxDimDataset(iMaterialistDataset):
    
    def load_image(self, img_num):
        """Load the specified image and return a [H,W,3] Numpy array.
        """
        # Given image_name (<string>.jpg), load the image into array
        info = self.image_info[img_num]
        width, height = info["width"], info["height"]
        raw_img = Image.open(info["path"]).convert("RGB")
        # Resize so the max dimension for each image is MAX_DIM in size
        ratio = min(MAX_DIM/info["width"], MAX_DIM/info["height"])
#         if ratio < 1.:
        resize_shape = (int(round(ratio*width)), int(round(ratio*height)))
        raw_img = raw_img.resize(resize_shape, 
                                 resample=PIL.Image.LANCZOS)
        img = np.asarray(raw_img)
        return img

    def load_mask(self, img_num):
        """Load instance masks for the given image.
        Different datasets use different ways to store masks. Override this
        method to load instance masks and return them in the form of an
        array of binary masks of shape [height, width, instances].
        
        Returns:
            masks: A bool array of shape [height, width, instance count] with
                a binary mask per instance.
            class_ids: a 1D array of class IDs of the instance masks.
        """
        info = self.image_info[img_num]
        width, height = info["width"], info["height"]
        ratio = min(MAX_DIM/width, MAX_DIM/height)
        masks = np.full((height, 
                         width, 
                         len(info["labels"])),
                        fill_value=0, 
                        dtype=np.uint8)
#         if ratio < 1.:
        resize_shape = int(round(ratio*width)), int(round(ratio*height))
        masks_resize = np.full((resize_shape[1], 
                                resize_shape[0], 
                                len(info["labels"])), 
                               fill_value=0, 
                               dtype=np.uint8)        
        category_ids = []
        for instance_num, category_annotation in enumerate(zip(info["labels"], info["annotations"])):
            category_id, encoded_pixels = category_annotation
            encoded_pixels = encoded_pixels.split(' ')
            mask = masks[:, :, instance_num]
            mask = self.RLE_to_submask(mask, encoded_pixels)
#             if ratio < 1.:
                #  mask = zoom(mask, ratio, order=1, mode='nearest')
            mask = zoom(mask, ratio)
            masks_resize[:, :, instance_num] = mask
#             else:
#                 masks[:, :, instance_num] = mask
            category_ids.append(int(category_id)+1) # Offset by 1 for BG at index 0.
#         if ratio < 1.:
#             masks = masks_resize
        category_ids = np.array(category_ids, dtype=np.int32)
        return masks_resize, category_ids


def main():
    parser = argparse.ArgumentParser(description='Process keyword arguments.')
    parser.add_argument('-n', '--new',
                        action='store_true',
                        help='''Start training from the beginning\n
                                (default: starts from last model)''')
    args = parser.parse_args()

    # INPUT_DIR = Path('/kaggle/input')
    # DATA_DIR = INPUT_DIR/"imaterialist-fashion-2019-FGVC6"
    # ROOT_DIR = Path('/kaggle/working')
    # INPUT_DIR = Path('/kaggle/input')
    # # get_ipython().system('pwd')
    DATA_DIR = Path('/home/chadwick/Documents/ImageSegmentation/iMaterialistFashion2019/imaterialist-fashion-2019-FGVC6')
    ROOT_DIR = Path('/home/chadwick/Documents/ImageSegmentation/iMaterialistFashion2019')
    COCO_WEIGHTS_PATH = 'mask_rcnn_coco.h5'
    # WEIGHTS_PATH = COCO_WEIGHTS_PATH
    MAX_DIM = 512

    # Actually, unfamiliar with how Kaggle handles their filesystem. Let's check.

    # print(os.listdir("/kaggle/"))
    # print(os.listdir("/kaggle/config"))
    # print(os.listdir(INPUT_DIR))
    # print(os.listdir(ROOT_DIR))
    # print(os.listdir(DATA_DIR))


    # ### Get the Matterport implementation of Mask R-CNN from github

    # # get_ipython().system('git clone https://www.github.com/matterport/Mask_RCNN.git')

    print("Current Directory: {}".format(os.curdir))
    print("Current Directory Contents: {}".format(os.listdir()))
    print("Changing dir...")
    os.chdir('Mask_RCNN')
    print("Current Directory: {}".format(os.curdir))
    print("Current Directory Contents: {}".format(os.listdir()))

    # get_ipython().system('rm -rf .git # to prevent an error when the kernel is committed')
    # get_ipython().system('rm -rf images assets # to prevent displaying images at the bottom of a kernel')


    ImageFile.LOAD_TRUNCATED_IMAGES = True
    COCO_WEIGHTS_PATH = 'mask_rcnn_coco.h5'
    # WEIGHTS_PATH = COCO_WEIGHTS_PATH
    
    train_df = pd.read_csv(DATA_DIR/"train.csv")

    print()
    print("Training df:")
    print(train_df.head())
    print()

    # Reading the json as a dict
    with open(DATA_DIR/"label_descriptions.json") as json_data:
        data = json.load(json_data)

    JSON_COLUMNS = list(data.keys())
    categories_df = json_normalize(data['categories'])
    # To be able to match this data with img_df data, rename 'id' column to 'ClassId'
    categories_df = categories_df.rename(columns={"id": "ClassId"})
    attributes_df = json_normalize(data['attributes'])
    info_df = json_normalize(data['info'])
    # Make a list matching up category string with id (as list index)
    category_names = list(categories_df["name"])

    print()
    print("Columns in info JSON:")
    print(JSON_COLUMNS)
    print()
    print("Categories df:")
    print(categories_df.head())
    print()
    print("Attributes df:")
    print(attributes_df.head())
    print()
    print("Info df:")
    print(info_df.head())
    print()
    print("Category Names, length:")
    print(category_names)
    print(len(category_names))


    # Let's remove the attributes from the ClassId column so 
    # that the dataset is simpler to deal with at first
    train_df["ClassId"] = train_df["ClassId"].apply(lambda x: x.split("_")[0])
    train_df_img_groups = train_df.groupby("ImageId")["EncodedPixels", "ClassId"].agg(lambda x: list(x))
    size_df = train_df.groupby("ImageId")["Height", "Width"].mean()
    train_df_img_groups = train_df_img_groups.join(size_df, on="ImageId")
    train_df_img_groups = train_df_img_groups.rename(columns={"ClassId": "CategoryIds"})


    # max_dim_data = iMaterialistMaxDimDataset(train_df_img_groups, category_names)
    # max_dim_data.prepare()


    # sample_dataset(max_dim_data)


    # # Let's train this muthasucka, vol. 2!
    # 
    # ![alt text](https://image-ticketfly.imgix.net/00/00/37/45/29-og.jpg "Bootsy Collins of Parliament! Check out that fashion though.")

    # This code partially supports k-fold training, 
    # you can specify the fold to train and the total number of folds here
    FOLD = 0
    N_FOLDS = 5
    # print(train_df_img_groups.iloc[0])
    train_img_group_labels = pd.DataFrame(train_df_img_groups['CategoryIds'])
    # print(train_img_group_labels.iloc[0])
    train_img_group_labels_array = np.array([np.array(v[0]) for v in train_img_group_labels.values])
    # print(train_img_group_labels_array)
    # print(train_img_group_labels_array[0])
    train_img_group_labels_binarized = MultiLabelBinarizer().fit_transform(train_img_group_labels_array)
    # print(train_img_group_labels_binarized[0])

    # train_fold, y_train, valid_fold, y_test =\
    #     iterative_train_test_split(train_df_img_groups, train_img_group_labels, test_size=0.2)
    # kf = KFold(n_splits=N_FOLDS, random_state=42, shuffle=True)
    # kf = StratifiedKFold(n_splits=N_FOLDS, random_state=42, shuffle=True)
    mkf = MultilabelStratifiedShuffleSplit(n_splits=N_FOLDS, test_size=0.2, random_state=42,)

    # ideally, this should be multilabel stratification
    # splits = kf.split(train_df_img_groups)
    splits = mkf.split(train_df_img_groups, train_img_group_labels_binarized)
    splits = list(splits)
    train_indices, valid_indices = splits[FOLD]
    # train_indices, valid_indices = train_indices[FOLD], valid_indices[FOLD]
    
    def img_to_categories(groups, indices):
        # train_groups = train_df_img_groups
        names = [groups.iloc[ind].name for ind in indices]
        category_ids = [groups.iloc[ind]['CategoryIds'] for ind in indices]
        groups_cat_names = [[category_names[int(_id)] for _id in img] for img in category_ids]
        names_categories = zip(names, groups_cat_names)
        return names_categories

    train_names_categories = img_to_categories(train_df_img_groups, train_indices)
    valid_names_categories = img_to_categories(train_df_img_groups, valid_indices)
    train_names, train_category_names = zip(*train_names_categories)
    valid_names, valid_category_names = zip(*valid_names_categories)
    train_categories = [train_df_img_groups.iloc[ind]['CategoryIds'] for ind in train_indices]
    valid_categories = [train_df_img_groups.iloc[ind]['CategoryIds'] for ind in valid_indices]

    def categories_to_hist(categories):
        flattened_categories = [int(cat) for group in categories for cat in group]
        hist = np.histogram(np.array(flattened_categories), bins=np.arange(len(category_names)))
        return hist

    def plot_categories_split(train_categories, valid_categories):
        fig, axes = plt.subplots(2, 1, figsize=(16,8))
        train_hist = categories_to_hist(train_categories)
        print(train_hist)
        axes[0].bar(train_hist[1][:-1], train_hist[0])
        valid_hist = categories_to_hist(valid_categories)
        print(valid_hist)
        axes[1].bar(valid_hist[1][:-1], valid_hist[0])
        plt.show()
        sys.exit()

    # print(list(splits))

    def get_fold(splits):
        for i, (train_index, valid_index) in enumerate(list(splits)):
            print(train_index)
            print(valid_index)
            if i == FOLD:
                return train_df_img_groups.iloc[train_index], train_df_img_groups.iloc[valid_index]

    folds = get_fold(splits)
    print(folds)
    train_fold, valid_fold = folds

    print("Training set has {} examples.".format(len(train_fold)))
    print("Validation set has {} examples.".format(len(valid_fold)))
    print()

    train_dataset = iMaterialistMaxDimDataset(train_fold, category_names)
    train_dataset.prepare()

    valid_dataset = iMaterialistMaxDimDataset(valid_fold, category_names)
    valid_dataset.prepare()

    # Data augmentation to be used
    augmentation = iaa.Sequential([
        iaa.Fliplr(0.5) # only horizontal flip here
    ])

    # augmentation = iaa.Sometimes(0.3, [
    #     iaa.Fliplr(0.5), # horizontal flips
    #     iaa.Crop(percent=(0, 0.1)), # random crops
    #     # Small gaussian blur with random sigma between 0 and 0.5.
    #     # But we only blur about 50% of all images.
    #     iaa.Sometimes(0.5,
    #         iaa.GaussianBlur(sigma=(0, 0.5))
    #     ),
    #     # Strengthen or weaken the contrast in each image.
    #     # iaa.ContrastNormalization((0.75, 1.5)),
    #     iaa.ContrastNormalization((0.75, 1.25)),
    #     # Add gaussian noise.
    #     # For 50% of all images, we sample the noise once per pixel.
    #     # For the other 50% of all images, we sample the noise per pixel AND
    #     # channel. This can change the color (not only brightness) of the
    #     # pixels.
    #     iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05*255), per_channel=0.5),
    #     # Make some images brighter and some darker.
    #     # In 20% of all cases, we sample the multiplier once per channel,
    #     # which can end up changing the color of the images.
    #     # iaa.Multiply((0.8, 1.2), per_channel=0.2),
    #     iaa.Multiply((0.9, 1.1), per_channel=0.2),
    #     # Apply affine transformations to each image.
    #     # Scale/zoom them, translate/move them, rotate them and shear them.
    #     # iaa.Affine(
    #     #     scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},
    #     #     translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)},
    #     #     rotate=(-25, 25),
    #     #     shear=(-8, 8)
    #     # )
    #     iaa.Affine(
    #         scale={"x": (0.9, 1.1), "y": (0.9, 1.1)},
    #         translate_percent={"x": (-0.1, 0.1), "y": (-0.1, 0.1)},
    #         rotate=(-25, 25),
    #         shear=(-4, 4)
    #     )
    # ])

    # augmentation = iaa.Sometimes(0.3, [
    #     iaa.Fliplr(0.5), # horizontal flips
    #     iaa.Crop(percent=(0, 0.1)), # random crops
    #     # Small gaussian blur with random sigma between 0 and 0.5.
    #     # But we only blur about 50% of all images.
    #     # iaa.Sometimes(0.5,
    #     iaa.Sometimes(0.3,
    #         iaa.GaussianBlur(sigma=(0, 0.5))
    #     ),
    #     # Strengthen or weaken the contrast in each image.
    #     # iaa.ContrastNormalization((0.75, 1.5)),
    #     iaa.ContrastNormalization((0.95, 1.05)),
    #     # Add gaussian noise.
    #     # For 50% of all images, we sample the noise once per pixel.
    #     # For the other 50% of all images, we sample the noise per pixel AND
    #     # channel. This can change the color (not only brightness) of the
    #     # pixels.
    #     # iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05*255), per_channel=0.5),
    #     iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05*255), per_channel=0.2),
    #     # Make some images brighter and some darker.
    #     # In 20% of all cases, we sample the multiplier once per channel,
    #     # which can end up changing the color of the images.
    #     # iaa.Multiply((0.8, 1.2), per_channel=0.2),
    #     iaa.Multiply((0.9, 1.1), per_channel=0.2),
    #     # Apply affine transformations to each image.
    #     # Scale/zoom them, translate/move them, rotate them and shear them.
    #     # iaa.Affine(
    #     #     scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},
    #     #     translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)},
    #     #     rotate=(-25, 25),
    #     #     shear=(-8, 8)
    #     # )
    #     iaa.Affine(
    #         scale={"x": (0.9, 1.1), "y": (0.9, 1.1)},
    #         translate_percent={"x": (-0.1, 0.1), "y": (-0.1, 0.1)},
    #         rotate=(-15, 15),
    #         # shear=(-2, 2)
    #     )
    # ])


    config = iMaterialistMaxDimConfig()

    # # Uncomment below if you want to test whether validation will run.
    # config.STEPS_PER_EPOCH = 1
    # print("Using {} steps per epoch for validation mem test".format(config.STEPS_PER_EPOCH))


    # Second go round. Let's increase epoch numbers (might break kernel with hangup)
    # As prescribed here https://medium.com/@connorandtrent/
    # mask-r-cnn-hyperparameter-experiments-with-weights-and-biases-bd2319faae26
    LRS = [1e-3, 5e-4, 2.5e-4]
    MOMS = [0.9, 0.95, 0.99]
    # LR = 1e-3
    # LR = 5e-4
    # LR = 1e-4
    # LR = 5e-5
    # LR = 1e-5
    EPOCHS = [25, 50, 110]

    config.EPOCHS = EPOCHS
    config.LRS = LRS
    config.MOMS = MOMS
    config.AUG = augmentation

    import warnings 
    warnings.filterwarnings("ignore")


    # INTERIM_WEIGHTS_PATH = '../fashion20190917T2204/mask_rcnn_fashion_0005.h5'
    # INTERIM_WEIGHTS_PATH = '../fashion20190922T1032/mask_rcnn_fashion_0009.h5'
    # INTERIM_WEIGHTS_PATH = '../fashion20191009T2021/mask_rcnn_fashion_0012.h5'
    # INTERIM_WEIGHTS_PATH = '../fashion20191011T1923/mask_rcnn_fashion_0019.h5'
    # INTERIM_WEIGHTS_PATH = '../fashion20191011T1923/mask_rcnn_fashion_0021.h5'
    # INTERIM_WEIGHTS_PATH = '../fashion20191018T2236/mask_rcnn_fashion_0007.h5'
    INTERIM_WEIGHTS_PATH = '../'
    model = modellib.MaskRCNN(mode='training', config=config, model_dir=INTERIM_WEIGHTS_PATH)
    # WEIGHTS_PATH = INTERIM_WEIGHTS_PATH
    if not args.new:
        INTERIM_WEIGHTS_PATH = model.find_last()
        print("INTERIM_WEIGHTS_PATH:")
        print(INTERIM_WEIGHTS_PATH)
        starting_epoch = int(INTERIM_WEIGHTS_PATH.split('_')[-1].split('.')[0])
    else:
        starting_epoch = 0
    print("Starting Epoch: {}".format(starting_epoch))
    EPOCHS = [e if e-starting_epoch > 0 else 0 for e in EPOCHS]
    # EPOCHS = [e-starting_epoch for e in EPOCHS if e-starting_epoch > 0]
    print("EPOCHS: {}".format(EPOCHS))
    # ONLY_INFERENCE = True
    # ONLY_INFERENCE = False
    # START_NEW = True

    if args.new:
        model.load_weights(COCO_WEIGHTS_PATH, 
                           by_name=True, 
                           exclude=['mrcnn_class_logits', 
                                    'mrcnn_bbox_fc', 
                                    'mrcnn_bbox', 
                                    'mrcnn_mask'])
    elif not args.new:
        model.load_weights(model.find_last(), 
                           by_name=True, 
                           exclude=['mrcnn_class_logits', 
                                    'mrcnn_bbox_fc', 
                                    'mrcnn_bbox', 
                                    'mrcnn_mask'])

    print("###########################################################################################################")
    print("# Model Layers/Parameters Summary:")
    model.keras_model.summary()
    config.save_to_file(model.log_dir)
    config.save_pickle(model.log_dir)

    # if not ONLY_INFERENCE:
    # Train the head branches\n
    # Passing layers="heads" freezes all layers except the head\n
    # layers. You can also pass a regular expression to select\n
    # which layers to train by name pattern.\n\n
    if len(EPOCHS) == 3 and EPOCHS[0] > 0:
        model.train(train_dataset,
                    valid_dataset,
                    learning_rate=LRS[0],
                    epochs=EPOCHS[0],
                    layers='heads')
    # history = model.keras_model.history.history
    # epochs = range(EPOCHS[0])
    # fig, axes = plt.subplots(1, 3, figsize=(16,8))
    # axes[0].plot(history['loss'], label="train loss")
    # axes[0].plot(history['val_loss'], label="valid loss")
    # axes[0].legend()
    # axes[1].plot(history['mrcnn_class_loss'], label="train class loss")
    # axes[1].plot(history['val_mrcnn_class_loss'], label="valid class loss")
    # axes[1].legend()
    # axes[2].plot(history['mrcnn_mask_loss'], label="train mask loss")
    # axes[2].plot(history['val_mrcnn_mask_loss'], label="valid mask loss")
    # axes[2].legend()
    # time.sleep(120)

    # Train the 3+ branches\n
    # Passing layers="3+" freezes all layers except those after\n
    # layer 2.\n\n
    if len(EPOCHS) >= 2 and EPOCHS[1] > 0:
        model.train(train_dataset,
                    valid_dataset,
                    learning_rate=LRS[1],
                    epochs=EPOCHS[1],
                    layers='all',
                    augmentation=augmentation
                    )
    # history = model.keras_model.history.history
    # epochs = range(EPOCHS[1])
    # fig, axes = plt.subplots(1, 3, figsize=(16,8))
    # axes[0].plot(history['loss'], label="train loss")
    # axes[0].plot(history['val_loss'], label="valid loss")
    # axes[0].legend()
    # axes[1].plot(history['mrcnn_class_loss'], label="train class loss")
    # axes[1].plot(history['val_mrcnn_class_loss'], label="valid class loss")
    # axes[1].legend()
    # axes[2].plot(history['mrcnn_mask_loss'], label="train mask loss")
    # axes[2].plot(history['val_mrcnn_mask_loss'], label="valid mask loss")
    # axes[2].legend()
    # time.sleep(120)

    # Train the  branches\n
    # Passing layers="heads" freezes all layers except the head\n
    # layers. You can also pass a regular expression to select\n
    # which layers to train by name pattern.\n\n
    if len(EPOCHS) >= 1 and EPOCHS[2] > 0:
        model.train(train_dataset,
                    valid_dataset,
                    learning_rate=LRS[2],
                    epochs=EPOCHS[2],
                    layers='all',
                    augmentation=augmentation
                    )

    # history = model.keras_model.history.history
    # epochs = range(EPOCHS[2])
    # fig, axes = plt.subplots(1, 3, figsize=(16,8))
    # axes[0].plot(history['loss'], label="train loss")
    # axes[0].plot(history['val_loss'], label="valid loss")
    # axes[0].legend()
    # axes[1].plot(history['mrcnn_class_loss'], label="train class loss")
    # axes[1].plot(history['val_mrcnn_class_loss'], label="valid class loss")
    # axes[1].legend()
    # axes[2].plot(history['mrcnn_mask_loss'], label="train mask loss")
    # axes[2].plot(history['val_mrcnn_mask_loss'], label="valid mask loss")
    # axes[2].legend()

    # plt.show()

    
if __name__ == '__main__':
    main()
