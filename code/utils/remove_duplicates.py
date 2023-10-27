import imagehash
import os
from PIL import Image

image_hashes = []

def remove_duplicates(image_path):
    '''
    Takes image, makes hash of image, checks if hash is in list, if not, adds hash to list, if already in list, deletes image
    '''
    # load image
    try:
        image = Image.open(image_path)
        # Make hash of image
        image_hash = imagehash.average_hash(image)
        # Check if hash is in list
        if image_hash in image_hashes:
            # Delete image
            os.remove(image_path)
        else:
            # Add hash to list
            image_hashes.append(image_hash)
    except Exception as e:
        print(e)
        os.remove(image_path)


if __name__ == '__main__':
    # get all images in directory and subdirectories ignoring other files
    images = os.walk('raw_data')
    for i,j,k in images:
        for image in k:
            if image.endswith('.jpg') or image.endswith('.png') or image.endswith('.jpeg'):
                remove_duplicates(os.path.join(i,image))
                # print(os.path.join(i,image))
