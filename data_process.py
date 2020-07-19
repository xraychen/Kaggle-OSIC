import os, cv2
import pydicom
import numpy as np

import matplotlib.pyplot as plt


def make_dir(file_path):
    dirname = os.path.dirname(file_path)
    try:
        if not os.path.exists(dirname): os.mkdir(dirname)
    except FileNotFoundError: pass


def normalize(pixel_array, image_size):
    pixel_array[pixel_array < 0] = 0
    pixel_array = cv2.resize(pixel_array, (image_size, image_size))
    pixel_array = pixel_array.astype(np.int16)
    return pixel_array


## TODO fix gdcm dependencies
def load_training_data(train_dir, train_csv, output_file, limit_num=50, image_size=512):

    users_id = os.listdir(train_dir)
    pad_image = np.zeros((image_size, image_size), np.int16)
    train_images = np.zeros((len(users_id), limit_num, image_size, image_size), np.int16)

    for i, user_id in enumerate(users_id):
        image_arr = os.listdir(os.path.join(train_dir, user_id))
        image_arr = [e.split('.')[0] for e in image_arr]
        image_arr = sorted([int(e) for e in image_arr])

        for j in range(limit_num):
            if j < len(image_arr):
                try:
                    image = pydicom.dcmread(os.path.join(train_dir, user_id, '{}.dcm'.format(image_arr[j])))
                    train_images[i, j, :, :] = normalize(image.pixel_array, image_size)
                except RuntimeError:
                    train_images[i, j, :, :] = pad_image
            else:
                train_images[i, j, :, :] = pad_image

    print(train_images.shape)

    make_dir(output_file)
    np.save(output_file, train_images)


def load_testing_data(test_dir, limit_num=50):
    pass


if __name__ == '__main__':
    load_training_data('raw/train', 'raw/train.csv', 'input/train_images.npy')