import pickle
import tensorflow as tf


def pad_resize_image(image, dims):
    image = tf.image.resize(
        image,
        dims,
        preserve_aspect_ratio=True
    )

    shape = tf.shape(image)

    sxd = dims[1] - shape[1]
    syd = dims[0] - shape[0]

    sx = tf.cast(
        sxd / 2,
        dtype=tf.int32
    )
    sy = tf.cast(
        syd / 2,
        dtype=tf.int32
    )

    paddings = tf.convert_to_tensor([
        [sy, syd - sy],
        [sx, sxd - sx],
        [0, 0]
    ])

    image = tf.pad(
        image,
        paddings,
        mode='CONSTANT',
        constant_values=128
    )
    return image


def preprocess_for_evaluation(image, image_size, dtype):
    image = pad_resize_image(
        image,
        [image_size, image_size]
    )

    image = tf.cast(image, dtype)
    image -= 128
    image /= 128
    return image


def arrange(subarray, predictions_arr, images_path_list):
    # subarrays = [([], []) for _ in range(10)]
    for prediction, image_path in zip(predictions_arr, images_path_list):
        index = int(100.0 * prediction // 10)
        subarray[index][0].append(100.0 * prediction)
        subarray[index][1].append(image_path)


def dump_bumble_results(subarray, dump_file, dump_results=True, print_results=True):
    if print_results:
        for i in range(len(subarray)):
            predictions, images_path = subarray[i]
            print("Between {}-{} : {} Images".format(i*10, (i+1)*10, len(predictions)))
            for i in range(len(predictions)):
                print('{:.2f}% - {}'.format(predictions[i], images_path[i]))
            print("")
    
    if dump_results:
        with open(dump_file, 'wb') as file:
            pickle.dump(subarray, file)


def dump_nsfw_results(subarray, dump_file, dump_results=True, print_results=True):
    if print_results:
        for i in range(len(subarray)):
            predictions, images_path, nsfw_category = subarray[i]
            print("Between {}-{} : {} Images".format(i*10, (i+1)*10, len(predictions)))
            for i in range(len(predictions)):
                print('{:.2f}% - {} - {}'.format(predictions[i], images_path[i], nsfw_category[i]))
            print("")
    
    if dump_results:
        with open(dump_file, 'wb') as file:
            pickle.dump(subarray, file)