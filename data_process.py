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
    # pixel_array = pixel_array.astype(np.int16)
    pixel_array = np.clip(pixel_array / 256, 0, 255)
    pixel_array = pixel_array.astype(np.uint8)
    return pixel_array


# TODO fix gdcm dependencies
def load_training_data(train_dir, train_csv, output_dir, limit_num=50, image_size=512):

    users_id = sorted(os.listdir(train_dir))
    # pad_image = np.zeros((image_size, image_size), np.int16)
    pad_image = np.zeros((image_size, image_size), np.uint8)

    # train_images = np.zeros((len(users_id), limit_num, image_size, image_size), np.int16)
    train_images = np.zeros((len(users_id), limit_num, image_size, image_size), np.uint8)
    train_x = np.zeros((len(users_id), 3), np.uint8)
    train_y = np.zeros((len(users_id), 146), np.int16)

    with open(train_csv) as f:
        content = f.read().splitlines()
        content = [e.split(',') for e in content]

    for i, user_id in enumerate(users_id):
        # train_images
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

        # trian_x
        temp = list(filter(lambda x: x[0] == user_id, content))

        train_x[i, :] = [
            int(temp[0][4]),
            ['Male', 'Female'].index(temp[0][5]),
            ['Never smoked', 'Ex-smoker', 'Currently smokes'].index(temp[0][6])
        ]

        # train_y
        p = 0
        for j in range(146):
            if j - 12 > int(temp[p][1]) and p < len(temp) - 1:
                p += 1
            train_y[i, j] = int(temp[p][2])

    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    np.save(os.path.join(output_dir, 'train_images.npy'), train_images)
    np.save(os.path.join(output_dir, 'train_x.npy'), train_x)
    np.save(os.path.join(output_dir, 'train_y.npy'), train_y)


def load_testing_data(test_dir, limit_num=50):
    pass


if __name__ == '__main__':
    load_training_data('raw/train', 'raw/train.csv', 'input/', limit_num=20, image_size=256)
