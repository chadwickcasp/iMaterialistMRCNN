import PIL
from PIL import Image, ImageFile
import matplotlib
import matplotlib.pyplot as plt
import os

MAX_DIM = 512
ImageFile.LOAD_TRUNCATED_IMAGES = True

if __name__ == '__main__':
    # for parent, dirs, files in os.walk('./imaterialist-fashion-2019-FGVC6/train/'):
    #     for i, file in enumerate(files):
    #         print("On file: {}, {}/{}".format(file, i+1, len(files)))
    #         # path = './imaterialist-fashion-2019-FGVC6/train/770bb4e147515d73c0e137f21973fba1 (copy).jpg'
    #         path = './imaterialist-fashion-2019-FGVC6/train/'+file
    #         try:
    #             raw_img = Image.open(path).convert("RGB")
    #         except OSError as e:
    #             print(e)
    #         width, height = raw_img.width, raw_img.height
    #         # Resize so the max dimension for each image is MAX_DIM in size
    #         ratio = min(MAX_DIM/width, MAX_DIM/height)
    #         if ratio < 1.:
    #             resize_shape = (int(round(ratio*width)), int(round(ratio*height)))
    #             raw_img = raw_img.resize(resize_shape, 
    #                                      resample=PIL.Image.LANCZOS)
    #         # Only uncomment this if you're SUPER sure of yourself
    #         raw_img.save(path)

    skipped_imgs = []
    for parent, dirs, files in os.walk('./imaterialist-fashion-2019-FGVC6/test/'):
        for i, file in enumerate(files):
            print("On file: {}, {}/{}".format(file, i+1, len(files)))
            # path = './imaterialist-fashion-2019-FGVC6/train/770bb4e147515d73c0e137f21973fba1 (copy).jpg'
            path = './imaterialist-fashion-2019-FGVC6/test/'+file
            try:
                raw_img = Image.open(path).convert("RGB")
            except OSError as e:
                print(e)
                skipped_imgs.append(path)
                continue
            width, height = raw_img.width, raw_img.height
            # Resize so the max dimension for each image is MAX_DIM in size
            ratio = min(MAX_DIM/width, MAX_DIM/height)
            if ratio < 1.:
                resize_shape = (int(round(ratio*width)), int(round(ratio*height)))
                raw_img = raw_img.resize(resize_shape, 
                                         resample=PIL.Image.LANCZOS)
            # Only uncomment this if you're SUPER sure of yourself
            raw_img.save(path)
    print(skipped_imgs)


    # path = './imaterialist-fashion-2019-FGVC6/train/fef7f887c41ce69ce198374fe4b93995.jpg'
    # try:
    #     raw_img = Image.open(path).convert("RGB")
    # except OSError as e:
    #     raw_img = Image.open(path)
    #     print(e)
    # width, height = raw_img.width, raw_img.height
    # # Resize so the max dimension for each image is MAX_DIM in size
    # ratio = min(MAX_DIM/width, MAX_DIM/height)
    # if ratio < 1.:
    #     resize_shape = (int(round(ratio*width)), int(round(ratio*height)))
    #     raw_img = raw_img.resize(resize_shape, 
    #                              resample=PIL.Image.LANCZOS)
    # # Only uncomment this if you're SUPER sure of yourself
    # raw_img.save(path)