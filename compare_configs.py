import pickle
from tkinter import Tk
from tkinter.filedialog import askopenfilename
from pathlib import Path
import sys

DATA_DIR = Path('/home/chadwick/Documents/ImageSegmentation/iMaterialistFashion2019/imaterialist-fashion-2019-FGVC6')
ROOT_DIR = Path('/home/chadwick/Documents/ImageSegmentation/iMaterialistFashion2019')
# Import Mask RCNN
sys.path.append(str(ROOT_DIR/'Mask_RCNN'))
print(sys.path)
from mrcnn.config import Config
from imaterialist_mrcnn import iMaterialistMaxDimConfig
Tk().withdraw() # we don't want a full GUI, so keep the root window from appearing

def main():
    config1 = iMaterialistMaxDimConfig()
    config2 = iMaterialistMaxDimConfig()

    filepath1 = askopenfilename() # show an "Open" dialog box and return the path to the selected file
    print(filepath1)
    filepath2 = askopenfilename() # show an "Open" dialog box and return the path to the selected file
    print(filepath2)

    config1.load_from_pickle(filepath1)
    config2.load_from_pickle(filepath2)
    
    config1_settings = vars(config1)
    config2_settings = vars(config2)
    print('\n'.join("%s: %s" % item for item in config1_settings.items() if item[0][:1] != '__'))
    print('\n'.join("%s: %s" % item for item in config2_settings.items() if item[0][:1] != '__'))
    print('\n'.join("%s" % item[0][:1] for item in config2_settings.items()))


if __name__ == '__main__':
    main()
