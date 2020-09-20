# iMaterialistMRCNN
This repo contains scripts used for training a Mask RCNN to recognize and segment clothing in an image. Datasets come from the Kaggle compeition: [Materialist (Fashion) 2019 at FGVC6.](https://www.kaggle.com/c/imaterialist-fashion-2019-FGVC6/overview)

## Dependencies & Structure
  * MRCNN trained on the COCO dataset to have a starting point. After cloning the repo, use the command:   
  `wget --quiet https://github.com/matterport/Mask_RCNN/releases/download/v2.0/mask_rcnn_coco.h5`  
    and put this in the `/Mask_RCNN` directory.
  * Data from the iMaterialist Fashion competition on Kaggle can be found [here.](https://www.kaggle.com/c/imaterialist-fashion-2019-FGVC6/data) Once downloaded, the data directory should be placed in this repo, with the following structure:  
```
.  
+-- imaterialist-fashion-2019-FGVC6    
|   +-- label_descriptions.csv  
|   +-- sample_submissions.csv  
|   +-- test  
|   |   +-- <.jpeg files>  
|   +-- train  
|   |   +-- <.jpeg files>  
```
  * Matterport's implementation of Mask RCNN can be found [here.](https://github.com/matterport/Mask_RCNN) I use a modified version of this in the `/Mask_RCNN` subdirectory here, but the link has the original source.
  
## Usage
To start training, download the files [here.](https://www.kaggle.com/c/imaterialist-fashion-2019-FGVC6/data) The data is part of an older Kaggle competition. Unzip the directory in the cloned repo and run `imaterialist_mrcnn.py` to start training.

## Notes/Mods to Source/Etc.
  * In `mrcnn.py-->MaskRCNN()-->train()`, the workers were halved because using all cores was crashing my PC.

## Sample Output
Left image is the training data, right is the result output from trained network. These images were obtained using the `inference_sample.py` script.
![Actor dude wearing a snazzy jacket.](https://raw.githubusercontent.com/chadwickcasp/iMaterialistMRCNN/master/samples/15861.png)
![Lady wearing hip skirt ruffle outfit.](https://raw.githubusercontent.com/chadwickcasp/iMaterialistMRCNN/master/samples/21432.png)
![Yeah...I know...that's not a headband.](https://raw.githubusercontent.com/chadwickcasp/iMaterialistMRCNN/master/samples/27603.png)
