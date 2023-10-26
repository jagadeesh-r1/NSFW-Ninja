# split images to dataset folder and divide into train and test folders
import os
import shutil
import random

def move_images():
    """Move images to train and test folders"""
    images = os.walk('raw_data')
    for i,j,k in images:
        for image in k:
            if image.endswith('.jpg') or image.endswith('.png') or image.endswith('.jpeg'):
                if random.random() < 0.8:
                    shutil.move(os.path.join(i,image), os.path.join('dataset/train',image))
                else:
                    shutil.move(os.path.join(i,image), os.path.join('dataset/test',image))

if __name__ == '__main__':
    move_images()