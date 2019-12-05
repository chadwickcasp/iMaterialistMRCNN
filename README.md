# iMaterialistMRCNN
This repo contains scripts used for training a Mask RCNN to recognize and segment clothing in an image. Datasets come from the Kaggle compeition: [Materialist (Fashion) 2019 at FGVC6.](https://www.kaggle.com/c/imaterialist-fashion-2019-FGVC6/overview)

## Dependencies & Structure
  * Data from the iMaterialist Fashion competition on Kaggle can be found [here.](https://www.kaggle.com/c/imaterialist-fashion-2019-FGVC6/data) Once downloaded, the data directory should be placed in this repo, with the following structore:
  ->imaterialist-fashion-2019-FGVC6    
    '->label_descriptions.csv  
    '->sample_submissions.csv  
    '->test  
      '->...<.jpg files>...  
    '->train  
      '->...<.jpg files>...  
      
  * Matterport's implementation of Mask RCNN can be found [here.](https://github.com/matterport/Mask_RCNN) I use a modified version of this in the ./Mask_RCNN subdirectory here, but the link has the original source.
  
## Usage
To start training, download the files [here.](https://www.kaggle.com/c/imaterialist-fashion-2019-FGVC6/data) The data is part of an older Kaggle competition. Unzip the directory in the cloned repo and run imaterialist_mrcnn.py to start training

## Notes/Mods to Source/Etc.

