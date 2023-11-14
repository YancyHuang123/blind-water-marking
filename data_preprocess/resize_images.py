# Import the necessary modules
import os
import glob
from PIL import Image

# Define the path where the images are stored
path = "/home/huangyanbin/0A__SoftwareProjects/Blind_watermark_DNN/datas/IntelImage"

# Loop through all the subfolders in the path
for folder in os.listdir(path):
    # Get the full path of the subfolder
    subfolder = os.path.join(path, folder)
    # Check if the subfolder is a directory
    if os.path.isdir(subfolder):
        # Loop through all the jpg images in the subfolder
        for subsubfolder in os.listdir(subfolder):
            subsubfolder = os.path.join(subfolder, subsubfolder)
            # Check if the sub-subfolder is a directory
            if os.path.isdir(subsubfolder):
                for s in os.listdir(subsubfolder):
                    s_path = os.path.join(subsubfolder, s)
                    # Check if the sub-subfolder is a directory
                    if os.path.isdir(s_path):
                        for image in glob.glob(s_path + "/*.jpg"):
                            # Open the image with PIL

                            img = Image.open(image)

                            img = img.resize((256, 256))
                            # Save the image in the same path with the same name
                            img.save(image)
