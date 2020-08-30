import os, cv2, pydicom, math, pickle, random, copy
import numpy as np
import matplotlib.pyplot as plt

from utils import *
from dcm_process import *


def fix_random_seed(random_seed):
    np.random.seed(random_seed)
    random.seed(random_seed)


def statistic(csv_file):
    with open(csv_file) as f:
        content = f.read().splitlines()[1:]
        content = [e.split(',') for e in content]

    for tag, idx in [('WEEK', 1), ('FVC', 2), ('PERCENT', 3), ('AGE', 4)]:
        data = [float(e[idx]) for e in content]
        print('{} = ({}, {})'.format(tag, np.mean(data), np.std(data)))


def onehot(idx, length):
    temp = np.zeros((length), np.float32)
    temp[idx] = 1
    return temp


def interpolate(c0, c1, x):
    return c0[1] + (c1[1] - c0[1]) * (x - c0[0]) / (c1[0] - c0[0])


def plot_images(images, values, output_file):
    fig ,axs = plt.subplots(10, 10, subplot_kw={'xticks': [], 'yticks': []}, figsize=(10, 10))
    for i in range(10):
        for j in range(10):
            try:
                axs[i][j].imshow(images[10 * i + j], cmap='gray')
                axs[i][j].set_title('{:.2f}'.format(values[10 * i + j]))
            except IndexError:
                pass

    plt.tight_layout()
    plt.subplots_adjust(wspace=0.5, hspace=0.5)
    plt.savefig(output_file, dpi=200)
    plt.close()


def get_info(user_row):
    line = user_row[0]
    info = np.concatenate([
        np.array([codec_a.encode(line[4])], np.float32),
        onehot(['Male', 'Female'].index(line[5]), 2),
        onehot(['Never smoked', 'Ex-smoker', 'Currently smokes'].index(line[6]), 3)
    ], axis=0).astype(np.float32)

    return info


def get_data(user_row, interpolation, add_noise, margin_week=12):
    data = []
    if interpolation:
        min_week = int(user_row[0][1])
        max_week = int(user_row[-1][1])

        k = 0
        for w in range(min_week, max_week + 1):
            flag = True
            line = user_row[k]
            prev = user_row[k - 1]
            if (w == int(line[1]) or w > int(line[1])) and k != len(user_row) - 1:
                week = float(w)
                fcv = float(line[2])
                percent = float(line[3])
                k += 1
            elif min(abs(w - int(prev[1])), abs(w - int(line[1]))) <= margin_week:
                week = float(w)
                fcv = interpolate((float(prev[1]), float(prev[2])), (float(line[1]), float(line[2])), w)
                percent = interpolate((float(prev[1]), float(prev[3])), (float(line[1]), float(line[3])), w)

                if add_noise:
                    # week += np.random.normal() * 12
                    fcv += np.random.normal() * 70
                    percent += np.random.normal() * 80
            else:
                flag = False

            if flag:
                # data.append([codec_w.encode(w), codec_f.encode(fcv), codec_p.encode(percent)])
                data.append([codec_w.encode(week), codec_f.encode(fcv), codec_p.encode(percent)])

    else:
        for line in user_row:
            data.append([codec_w.encode(line[1]), codec_f.encode(line[2]), codec_p.encode(line[3])])

    data = np.array(data, np.float32)

    return data


def process_dcm(dcm, image_size, window_width=-1500, window_center=-600):
    try:
        image = transform_ctdata(dcm, window_width, window_center)

        if np.mean(image) < 50 or np.mean(image) > 170:
            print('skip')
        else:
            image = get_lung_img(image)
            image = get_square_img(image)

        image = cv2.resize(image, (image_size, image_size))

    except RuntimeError as e:
        print(e)
        image = np.zeros((image_size, image_size), np.uint8)

    return image


def get_images(user_id, image_dir, limit_num=20, image_size=256):
    images = []
    empty_image = np.zeros((image_size, image_size), np.uint8)

    files = os.listdir(os.path.join(image_dir, user_id))
    files = sorted([int(f.split('.')[0]) for f in files])
    files = ['{}.dcm'.format(f) for f in files]

    if limit_num == 1:
        file = files[math.floor(len(files) / 2)]
        dcm = pydicom.dcmread(os.path.join(image_dir, user_id, file))
        dcm = process_dcm(dcm, image_size)
        images.append(dcm)

    elif len(files) < limit_num:
        for i in range(limit_num):
            if i < len(files):
                file = files[i]
                dcm = pydicom.dcmread(os.path.join(image_dir, user_id, file))
                dcm = process_dcm(dcm, image_size)
                images.append(dcm)
            else:
                images.append(empty_image)

    else:
        k = len(files) / limit_num
        for i in range(limit_num):
            file = files[math.floor(i * k)]
            dcm = pydicom.dcmread(os.path.join(image_dir, user_id, file))
            dcm = process_dcm(dcm, image_size)
            images.append(dcm)

    return images


def process_data(csv_file, image_dir=None, output_file=None, interpolation=True, add_noise=True, limit_num=20, image_size=256):
    with open(csv_file) as f:
        content = f.read().splitlines()[1:]
        content = [e.split(',') for e in content]

    output = []
    users_id = sorted(list(set([line[0] for line in content])))

    for i, user_id in enumerate(users_id):
        # print('{} {}/{}'.format(user_id, i, len(users_id)))

        user_row = list(filter(lambda x: x[0] == user_id, content))
        temp = {
            'id': user_id,
            'info': get_info(user_row),
            'data': get_data(user_row, interpolation, add_noise),
            'images': get_images(user_id, image_dir, limit_num, image_size) if image_dir is not None else None
        }
        output.append(temp)

    if output_file is not None:
        with open(output_file, 'wb') as f:
            pickle.dump(output, f)
    else:
        return output


if __name__ == '__main__':
    fix_random_seed(42)
    # statistic('raw/train.csv')
    process_data('raw/train.csv', 'raw/train', limit_num=1)
