import argparse
import yaml
import os
from typing import List
import random
import glob
from tqdm import tqdm
import imghdr

import tensorflow as tf
from absl import logging as absl_logging
import utils

from utils import preprocess_for_evaluation


def read_image(filename: str) -> tf.Tensor:
    image = tf.io.read_file(filename)
    image = tf.io.decode_jpeg(image, channels=3)

    image = preprocess_for_evaluation(
        image,
        480,
        tf.float16
    )

    image = tf.reshape(image, -1)

    return image


def inferencev1(args) -> None:
    images_path_list = glob.glob(args['dataset_path'])
    if args['num_images'] is not None:
        num_images = min(args['num_images'], len(images_path_list))
        if args['pick_random']:
            images_path_list = random.sample(images_path_list, num_images)
        else:
            images_path_list = images_path_list[:num_images]

    model = tf.saved_model.load(args['bumble_model_folder'])

    for image_path in images_path_list:
        image = read_image(image_path)
        preds = model([image])
        print(f'{100 * tf.get_static_value(preds[0])[0]:.2f}% - {image_path}')


def inferencev2(args) -> None:
    images_path_list = glob.glob(args['dataset_path'])
    
    if args['num_images'] is not None:
        num_images = min(args['num_images'], len(images_path_list))
        if args['pick_random']:
            images_path_list = random.sample(images_path_list, num_images)
        else:
            images_path_list = images_path_list[:num_images]
    
    model = tf.saved_model.load(args['bumble_model_folder'])
    
    batch_size = args['batch_size']
    total_images = len(images_path_list)
    num_batches = len(images_path_list) // batch_size

    print("TOTAL IMAGES : {}".format(total_images))

    if num_batches == 0 and len(images_path_list) != 0:
        num_batches = 1

    pred_path = [([], []) for _ in range(10)]

    ###### Inference ########
    for i in tqdm(range(num_batches)):

        imgs_list = []
        filtered_images_path = []

        for j in range(batch_size):
            if i*batch_size + j < total_images:
                img_path = images_path_list[i*batch_size + j]
                file_type = imghdr.what(img_path)
                if file_type in args['allowed_file_types'] or file_type is None:
                    imgs_list.append(read_image(img_path))
                    filtered_images_path.append(img_path)
                # else:
                    # print("Bad Image : {}".format(img_path))

        preds = model(imgs_list)
        preds = tf.get_static_value(preds)
        utils.arrange(pred_path, preds[:,0], filtered_images_path)

    utils.dump_bumble_results(pred_path, args['dump_file'], args['dump_results'], args['print_results'])


if __name__ == '__main__':
    tf.get_logger().setLevel('ERROR')
    absl_logging.set_verbosity(absl_logging.ERROR)

    with open('configs/bumble_config.yaml') as f:
        config = yaml.safe_load(f)

    inferencev2(config)
