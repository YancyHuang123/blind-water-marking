# Import the necessary modules
from multiprocessing.dummy import Namespace
import os
import glob
from tkinter.font import names
from PIL import Image

width=256

# Define the path where the images are stored
path = "/home/huangyanbin/0A__SoftwareProjects/Blind_watermark_DNN/datas/IntelImage"
names=['seg_test','seg_train']
s_path=f'{path}/seg_pred/seg_pred'

for image in glob.glob(s_path + "/*.jpg"):
        # Open the image with PIL
        img = Image.open(image)

        img = img.resize((width,width))
        # Save the image in the same path with the same name
        img.save(image)

for n in names:
    s_path=f'{path}/{n}/{n}'
    for folder in os.listdir(s_path):
        subfolder = os.path.join(s_path, folder)
        for image in glob.glob(subfolder + "/*.jpg"):
            # Open the image with PIL
            img = Image.open(image)

            img = img.resize((width,width))
            # Save the image in the same path with the same name
            img.save(image)
