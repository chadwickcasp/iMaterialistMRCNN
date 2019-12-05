import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from pathlib import Path
import os
import sys
# !pip install tensorflow==2.0.0-beta1
import tensorflow as tf
import json
import multiprocessing
from tkinter import Tk
from tkinter.filedialog import askopenfilename

Tk().withdraw() # we don't want a full GUI, so keep the root window from appearing

from tensorflow.python.client import device_lib

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

DATA_DIR = Path('/home/chadwick/Documents/ImageSegmentation/iMaterialistFashion2019/imaterialist-fashion-2019-FGVC6')
ROOT_DIR = Path('/home/chadwick/Documents/ImageSegmentation/iMaterialistFashion2019')

# Import Mask RCNN
sys.path.append(str(ROOT_DIR/'Mask_RCNN'))
print(sys.path)
from mrcnn.config import Config
from mrcnn import utils
import mrcnn.model as modellib
from mrcnn import visualize
from mrcnn.model import log
from imaterialist_mrcnn import iMaterialistMaxDimDataset, \
  iMaterialistMaxDimConfig

import json
from pandas.io.json import json_normalize

# os.chdir('Mask_RCNN')
train_df = pd.read_csv(DATA_DIR/"train.csv")

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

# Let's remove the attributes from the ClassId column so 
# that the dataset is simpler to deal with at first
train_df["ClassId"] = train_df["ClassId"].apply(lambda x: x.split("_")[0])
train_df_img_groups = train_df.groupby("ImageId")["EncodedPixels", "ClassId"].agg(lambda x: list(x))
size_df = train_df.groupby("ImageId")["Height", "Width"].mean()
train_df_img_groups = train_df_img_groups.join(size_df, on="ImageId")
train_df_img_groups = train_df_img_groups.rename(columns={"ClassId": "CategoryIds"})
train_dataset = iMaterialistMaxDimDataset(train_df_img_groups, category_names)
train_dataset.prepare()

# Now sample a few validation images, to get a sense of how well
# the network is training
class iMaterialistMaxDimInferenceConfig(Config):
    def __init__(self):
        self.GPU_COUNT = 1
        self.IMAGES_PER_GPU = 2

filename = askopenfilename() # show an "Open" dialog box and return the path to the selected file
print(filename)
TRAINED_WEIGHTS_PATH = filename
LOAD_PATH = '/'.join(TRAINED_WEIGHTS_PATH.split('/')[:-1])
inference_config = iMaterialistMaxDimInferenceConfig()
inference_config.load_from_pickle(LOAD_PATH)

model = modellib.MaskRCNN(mode='inference', 
                          config=inference_config, 
                          model_dir=ROOT_DIR)

model.load_weights(TRAINED_WEIGHTS_PATH, 
                   by_name=True)

# Test on N random images
image_gt_data = []
image_ids = []
N = 10
for i in range(N):
    image_id = random.choice(train_dataset.image_ids)
    image_ids.append(image_id)
    original_image, image_meta, gt_class_id, gt_bbox, gt_mask =\
        modellib.load_image_gt(train_dataset, inference_config, 
                               image_id, use_mini_mask=False)
    log("original_image", original_image)
    log("image_meta", image_meta)
    log("gt_class_id", gt_class_id)
    log("gt_bbox", gt_bbox)
    log("gt_mask", gt_mask)
    image_gt_data.append((original_image, image_meta, gt_class_id, gt_bbox, gt_mask))

original_images, _, _, _, _ = zip(*image_gt_data)
batch_len = inference_config.IMAGES_PER_GPU
batch_indexes = range(len(original_images))[::batch_len]
print(batch_indexes)
results = []
for i in batch_indexes:
    batch_images = original_images[i:i+batch_len]
    results += model.detect(batch_images, verbose=1)

for i, gt_data in enumerate(image_gt_data):
    original_image, image_meta, gt_class_id, gt_bbox, gt_mask = gt_data
    fig, ax = plt.subplots(1, 2)
    fig.canvas.set_window_title(str(image_ids[i]))
    visualize.display_instances(original_image, gt_bbox, gt_mask, gt_class_id, 
                                train_dataset.class_names, figsize=(8, 8), ax=ax[0])
    r = results[i]
    visualize.display_instances(original_image, r['rois'], r['masks'], r['class_ids'], 
                                train_dataset.class_names, r['scores'], ax=ax[1])
    fig.tight_layout()
plt.show()
