import os, cv2, pickle
import pydicom
import numpy as np
import matplotlib.pyplot as plt

from utils import *


def make_dir(file_path):
    dirname = os.path.dirname(file_path)
    try:
        if not os.path.exists(dirname):
            os.mkdir(dirname)
    except FileNotFoundError:
        pass


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
    temp = np.zeros((length), np.float32)
    temp[idx] = 1
    return temp


def to_onehot(value, category):
    if (category == 'week'):
        idx = int(value) - Y_OFFSET
        length = Y_LENGTH
    elif (category == 'age'):
        idx = (int(value) - 50) // 5
        idx = np.clip(idx, 0, 7)
        length = 8
    elif (category == 'sex'):
        idx = ['Male', 'Female'].index(value)
        length = 2
    elif (category == 'smoking_status'):
        idx = ['Never smoked', 'Ex-smoker', 'Currently smokes'].index(value)
        length = 3
    else:
        raise KeyError

    return onehot(idx, length)


# def codec_fcv(value, decode=False):
#     MEAN, STD = 2690.479018721756, 832.5021066817238

#     if type(value) == str:
#         value = float(value)

#     if decode:
#         return (value * STD) + MEAN
#         # return value * 4000.
#     else:
#         return (value - MEAN) / STD
#         # return value / 4000.


# def codec_percent(value, decode=False):
#     MEAN, STD = 77.67265350296326, 19.81686156299212

#     if type(value) == str:
#         value = float(value)

#     if decode:
#         return (value * STD) + MEAN
#         # return value * 100.
#     else:
#         return (value - MEAN) / STD
#         # return value / 100.


def normalize(pixel_array, image_size):
    pixel_array[pixel_array < 0] = 0
    pixel_array = cv2.resize(pixel_array, (image_size, image_size))
    if (np.max(pixel_array) > 0):
        pixel_array = np.clip((pixel_array / np.max(pixel_array)) * 255, 0, 255)
    pixel_array = pixel_array.astype(np.uint8)
    return pixel_array


def process_data(csv_file, image_dir, limit_num=20, image_size=256, return_y=True):
    with open(csv_file) as f:
        content = f.read().splitlines()[1:]
        content = [e.split(',') for e in content]

    x = np.zeros((len(content), X_LENGTH), np.float32)
    y = np.zeros((len(content), Y_LENGTH), np.float32)

    cache_user_id = None
    cache_y = np.zeros((Y_LENGTH), np.float32)
    cache_image = np.zeros((limit_num, image_size, image_size), np.uint8)

    empty_image = np.zeros((image_size, image_size), np.uint8)
    images = []

    images_id = np.zeros((len(content)), np.int16)

    for i, line in enumerate(content):
        # generate x
        x[i, :] = np.concatenate((
            np.array([codec_f.encode(line[2]), codec_p.encode(line[3])], np.float32),
            to_onehot(line[1], 'week'),
            to_onehot(line[4], 'age'),
            to_onehot(line[5], 'sex'),
            to_onehot(line[6], 'smoking_status')
        ), axis=0)

        user_id = line[0]
        if cache_user_id != user_id:
            cache_user_id = user_id
            # generate y
            if return_y:
                user_filter = list(filter(lambda x: x[0] == user_id, content))
                p = 0
                for j in range(Y_LENGTH):
                    if j + Y_OFFSET > int(user_filter[p][1]) and p < len(user_filter) - 1:
                        p += 1
                    cache_y[j] = codec_f.encode(user_filter[p][2])

            # generate image
            image_arr = os.listdir(os.path.join(image_dir, user_id))
            image_arr = [e.split('.')[0] for e in image_arr]
            image_arr = sorted([int(e) for e in image_arr])

            for j in range(limit_num):
                if j < len(image_arr):
                    try:
                        if limit_num == 1:
                            cache_image[j, :, :] = empty_image
                        else:
                            image = pydicom.dcmread(os.path.join(image_dir, user_id, '{}.dcm'.format(image_arr[j])))
                            cache_image[j, :, :] = normalize(image.pixel_array, image_size)
                    except RuntimeError:
                        cache_image[j, :, :] = empty_image
                else:
                    cache_image[j, :, :] = empty_image

            images.append(cache_image)
            image_id = len(images) - 1

        y[i, :] = cache_y
        images_id[i] = image_id

    images = np.array(images, np.uint8)

    if return_y:
        return images, images_id, x, y
    else:
        return images, images_id, x


def statistic():
    with open('raw/train.csv') as f:
        content = f.read().splitlines()[1:]
        content = [e.split(',') for e in content]

    row_01 = [float(e[2]) for e in content]
    row_02 = [float(e[3]) for e in content]

    print('fcv:     {}, {}'.format(np.mean(row_01), np.std(row_01)))
    print('percent: {}, {}'.format(np.mean(row_02), np.std(row_02)))


def process_training_data():
    images, images_id, x, y = process_data('raw/train.csv', 'raw/train', limit_num=1)

    with open('input/train_new.pickle', 'wb') as f:
        pickle.dump((images, images_id, x, y), f)


def process_testing_data():
    images, images_id, x = process_data('raw/test.csv', 'raw/test', limit_num=1, return_y=False)

    with open('input/test.pickle', 'wb') as f:
        pickle.dump((images, images_id, x), f)


# X_LENGTH = 161
# Y_LENGTH = 146
# Y_OFFSET = -12

# codec_f = Codec(tag='fcv')
# codec_p = Codec(tag='percent')


if __name__ == '__main__':
    process_training_data()
    process_testing_data()
    # statistic()
