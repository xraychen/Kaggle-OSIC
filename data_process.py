import os, cv2
import pydicom
import numpy as np

import matplotlib.pyplot as plt
from pydicom.pixel_data_handlers.util import apply_color_lut


def make_dir(file_path):
    dirname = os.path.dirname(file_path)
    try:
        if not os.path.exists(dirname): os.mkdir(dirname)
    except FileNotFoundError: pass


def plot_images(images, output_file):
    fig ,axs = plt.subplots(10, 10, subplot_kw={'xticks': [], 'yticks': []}, figsize=(10, 10))
    for i in range(10):
        for j in range(10):
            axs[i][j].imshow(images[10 * i + j], cmap='gray')

    plt.tight_layout()
    plt.subplots_adjust(wspace=0, hspace=0)
    plt.savefig(output_file, dpi=150)
    plt.close()


def onehot(idx, length):
    temp = np.zeros((length), np.uint8)
    temp[idx] = 1
    return temp


def to_onehot(value, category):
    if (category == 'age'):
        idx = (int(value) - 50) // 5
        idx = np.clip(idx, 0, 7)
        length = 8
    elif (category == 'sex'):
        idx = ['Male', 'Female'].index(value)
        length = 2
    elif (category == 'smoke_status'):
        idx = ['Never smoked', 'Ex-smoker', 'Currently smokes'].index(value)
        length = 3
    else:
        raise KeyError

    return onehot(idx, length)


def normalize(pixel_array, image_size):
    pixel_array[pixel_array < 0] = 0
    pixel_array = cv2.resize(pixel_array, (image_size, image_size))
    if (np.max(pixel_array) > 0):
        pixel_array = np.clip((pixel_array / np.max(pixel_array)) * 255, 0, 255)
    pixel_array = pixel_array.astype(np.uint8)
    return pixel_array


# TODO fix gdcm dependencies
def load_training_data(train_dir, train_csv, output_dir, limit_num=50, image_size=512):
    users_id = sorted(os.listdir(train_dir))
    # pad_image = np.zeros((image_size, image_size), np.uint8)
    pad_image = np.zeros((image_size, image_size, 3), np.uint8)

    # train_images = np.zeros((len(users_id), limit_num, image_size, image_size), np.uint8)
    train_images = np.zeros((len(users_id), limit_num, image_size, image_size, 3), np.uint8)
    train_x = np.zeros((len(users_id), 13), np.uint8)
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




                    pixel_array = apply_color_lut(image.pixel_array, palette='HOT_IRON')
                    pixel_array = cv2.resize(pixel_array, (image_size, image_size))
                    train_images[i, j, :, :, :] = pixel_array
                    # train_images[i, j, :, :] = normalize(image.pixel_array, image_size)
                except RuntimeError:
                    # train_images[i, j, :, :] = pad_image
                    train_images[i, j, :, :, :] = pad_image
            else:
                # train_images[i, j, :, :] = pad_image
                train_images[i, j, :, :, :] = pad_image

        # trian_x
        temp = list(filter(lambda x: x[0] == user_id, content))

        train_x[i, :] = np.concatenate((
            to_onehot(temp[0][4], 'age'),
            to_onehot(temp[0][5], 'sex'),
            to_onehot(temp[0][6], 'smoke_status')
        ), axis=0)

        # train_y
        p = 0
        for j in range(146):
            if j - 12 > int(temp[p][1]) and p < len(temp) - 1:
                p += 1
            train_y[i, j] = int(temp[p][2])

    plot_images(train_images[:, 0, :, :], 'plot/sample_hot_iron.png')
    # print(np.mean(train_y), np.std(train_y))

    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    # np.save(os.path.join(output_dir, 'train_images.npy'), train_images)
    # np.save(os.path.join(output_dir, 'train_x.npy'), train_x)
    # np.save(os.path.join(output_dir, 'train_y.npy'), train_y)


def load_testing_data(test_dir, limit_num=50):
    pass


def test():
    image = pydicom.dcmread('raw/train/ID00007637202177411956430/1.dcm')
    print(image)
    return 0
    pixel_array = image.pixel_array
    print(pixel_array.dtype)
    pixel_array = apply_color_lut(pixel_array, palette='HOT_IRON')
    print(pixel_array.dtype)
    plt.imshow(pixel_array)
    plt.savefig('plot/_.png')
    plt.close()


if __name__ == '__main__':
    # load_training_data('raw/train', 'raw/train.csv', 'input/', limit_num=10, image_size=256)
    test()
