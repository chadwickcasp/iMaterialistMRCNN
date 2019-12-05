import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from pathlib import Path
import os
import sys
# !pip install tensorflow==2.0.0-beta1
import tensorflow as tf
import json
import multiprocessing
from tqdm import tqdm
from tkinter import Tk
from tkinter.filedialog import askopenfilename
import pickle

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
from mrcnn.utils import compute_ap_range, compute_ap
from imaterialist_mrcnn import iMaterialistMaxDimDataset, \
  iMaterialistMaxDimConfig

import json
from pandas.io.json import json_normalize
import warnings


def get_fold(train_df, splits, fold):
    for i, (train_index, valid_index) in enumerate(list(splits)):
        print(train_index)
        print(valid_index)
        if i == fold:
            return train_df.iloc[train_index], train_df.iloc[valid_index]



# Now sample a few validation images, to get a sense of how well
# the network is training
class iMaterialistMaxDimInferenceConfig(Config):
    def __init__(self):
        self.GPU_COUNT = 1
        self.IMAGES_PER_GPU = 2

def save_pickle(obj, filepath, filename):
    try:
        os.makedirs(filepath)
    except FileExistsError:
        # directory already exists
        pass

    if not os.path.exists(filepath): 
        open(os.path.join(filepath, filename), 'w')

    with open(os.path.join(filepath, filename), 'wb') as f:
        pickle.dump(obj, f)

def compute_overall_mAP(model, image_ids, dataset, inference_config, filepath, iou_thresholds=None):
    warnings.filterwarnings("ignore")

    image_gt_data = []
    overall_mAP = []
    overall_precisions = []
    overall_recalls = []
    overall_overlaps = []
    batch_len = inference_config.IMAGES_PER_GPU

    for i, image_id in tqdm(list(enumerate(image_ids)), ascii=True, desc="Loading GT data..."):
        original_image, image_meta, gt_class_id, gt_bbox, gt_mask =\
            modellib.load_image_gt(dataset, inference_config, 
                                   image_id, use_mini_mask=False)
        image_gt_data.append((original_image, image_meta, gt_class_id, gt_bbox, gt_mask))

        if (i+1) % batch_len == 0:
            original_images, image_metas, gt_class_ids, gt_boxes, gt_masks = zip(*image_gt_data)
            batch_results = model.detect(original_images)
            pred_boxes = [result['rois'] for result in batch_results]
            pred_class_ids = [result['class_ids'] for result in batch_results]
            pred_scores = [result['scores'] for result in batch_results]
            pred_masks = [result['masks'] for result in batch_results]
            img_data = zip(gt_boxes, gt_class_ids, gt_masks, pred_boxes, pred_class_ids, pred_scores, pred_masks)

            for data in img_data:
                gt_boxes, gt_class_ids, gt_masks, pred_boxes, pred_class_ids, pred_scores, pred_masks = data
                mAP, precisions, recalls, overlaps = compute_ap(gt_boxes, gt_class_ids, gt_masks,
                                                                pred_boxes, pred_class_ids, pred_scores, pred_masks,
                                                                iou_threshold=0.5)
                overall_mAP.append(mAP)
                overall_precisions.append(precisions)
                overall_recalls.append(recalls)
                overall_overlaps.append(overlaps)
            image_gt_data = []
            save_pickle(zip(overall_mAP, 
                            overall_precisions, 
                            overall_recalls, 
                            overall_overlaps), 
                        filepath, 
                        "acc_tmp.pickle")

    warnings.filterwarnings("default")

    return overall_mAP, overall_precisions, overall_recalls, overall_overlaps

def main():
    #######################################################################################################
    # Load data
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
    # print(list(splits))
    folds = get_fold(train_df_img_groups, splits, FOLD)
    print(folds)
    train_fold, valid_fold = folds

    print("Training set has {} examples.".format(len(train_fold)))
    print("Validation set has {} examples.".format(len(valid_fold)))
    print()

    # Test on a small subset of the data
    # train_fold = train_fold[:6]
    # valid_fold = valid_fold[:6]

    train_dataset = iMaterialistMaxDimDataset(train_fold, category_names)
    train_dataset.prepare()

    valid_dataset = iMaterialistMaxDimDataset(valid_fold, category_names)
    valid_dataset.prepare()

    # show an "Open" dialog box and return the path to the selected file
    filename = askopenfilename(title="Select an .h5 file to load into the inference model") 
    print(filename)
    TRAINED_WEIGHTS_PATH = filename
    MODEL_FILE = TRAINED_WEIGHTS_PATH.split('/')[-1].split('.')[0]
    LOAD_PATH = '/'.join(TRAINED_WEIGHTS_PATH.split('/')[:-1])
    inference_config = iMaterialistMaxDimInferenceConfig()
    inference_config.load_from_pickle(LOAD_PATH)

    model = modellib.MaskRCNN(mode='inference', 
                              config=inference_config, 
                              model_dir=ROOT_DIR)

    model.load_weights(TRAINED_WEIGHTS_PATH, 
                       by_name=True)

    ###################################################################################################################
    # TODO: Compute average precision for training and test sets.
    # TODO: Show progress through the data somehow (tqdm?)
    # TODO: Print this through CLI, but also save file tagged with partcular h5 file
    # Test on N random images

    image_ids = train_dataset.image_ids
    val_image_ids = valid_dataset.image_ids
    print(len(image_ids))
    print(len(val_image_ids))
    print(image_ids)
    print(val_image_ids)

    train_aps, train_recalls, train_precisions, train_overlaps = \
      compute_overall_mAP(model, image_ids, train_dataset, inference_config, LOAD_PATH)
    valid_aps, valid_recalls, valid_precisions, valid_overlaps = \
      compute_overall_mAP(model, val_image_ids, valid_dataset, inference_config, LOAD_PATH)

    save_pickle(zip(train_aps, 
                    train_recalls, 
                    train_precisions, 
                    train_overlaps), 
                LOAD_PATH, 
                MODEL_FILE.split('_')[-1]+"_train.pickle")
    save_pickle(zip(valid_aps, 
                    valid_recalls, 
                    valid_precisions, 
                    valid_overlaps), 
                LOAD_PATH, 
                MODEL_FILE.split('_')[-1]+"_valid.pickle")
    print(train_aps)
    print(valid_aps)


if __name__ == '__main__':
    main()



