# This notebook is a copy of all files for submission in kaggle.

import os, cv2, pydicom, math, pickle, random, copy, time, sys
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset


# TEST_CSV = '../input/osic-pulmonary-fibrosis-progression/test.csv'
# TEST_DIR = '../input/osic-pulmonary-fibrosis-progression/test'
# SUBMIT_CSV = 'submission.csv'
# MODEL_FILE = ''

TEST_CSV = 'raw/test.csv'
TEST_DIR = 'raw/test'
SUBMIT_CSV = 'output/notebook.csv'
MODEL_FILE = 'model/lsm_12.npy'

NUM_WORKERS = 6


# utils.py
WEEK    = (31.861846352485475, 23.240045178171002)
FVC     = (2690.479018721756, 832.5021066817238)
PERCENT = (77.67265350296326, 19.81686156299212)
AGE     = (67.18850871530019, 7.055116199848975)
IMAGE   = (615.48615, 483.8854)

def make_dir(file_path):
    dirname = os.path.dirname(file_path)
    try:
        if not os.path.exists(dirname):
            os.mkdir(dirname)
    except FileNotFoundError:
        pass


class Codec:
    def __init__(self, tag='fvc'):
        if tag == 'week':
            self.mean, self.std = WEEK
        elif tag == 'fvc':
            self.mean, self.std = FVC
        elif tag == 'percent':
            self.mean, self.std = PERCENT
        elif tag == 'age':
            self.mean, self.std = AGE
        elif tag == 'image':
            self.mean, self.std = IMAGE
        else:
            raise KeyError

    def encode(self, value, scale_only=False):
        value = float(value) if type(value) == str else value
        if scale_only:
            return value / self.std
        else:
            return (value - self.mean) / self.std

    def decode(self, value, scale_only=False):
        value = float(value) if type(value) == str else value
        if scale_only:
            return value * self.std
        else:
            return value * self.std + self.mean


codec_w = Codec(tag='week')
codec_f = Codec(tag='fvc')
codec_p = Codec(tag='percent')
codec_a = Codec(tag='age')
codec_i = Codec(tag='image')


# data_process.py
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
                fcv = float(line[2])
                percent = float(line[3])
                k += 1
            elif min(abs(w - int(prev[1])), abs(w - int(line[1]))) <= margin_week:
                fcv = interpolate((float(prev[1]), float(prev[2])), (float(line[1]), float(line[2])), w)
                percent = interpolate((float(prev[1]), float(prev[3])), (float(line[1]), float(line[3])), w)

            else:
                flag = False

            if add_noise:
                # fcv += np.random.normal() * 15
                # percent += np.random.normal() * 5
                fcv += np.random.normal() * 70
                percent += np.random.normal() * 10
                pass

            if flag:
                data.append([codec_w.encode(w), codec_f.encode(fcv), codec_p.encode(percent)])

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


# train_lsm.py
class LsmDataset(Dataset):
    def __init__(self, bundle, features=None, tag='train'):
        self.x = []
        self.y = []

        if tag == 'train':
            for k, item in enumerate(bundle):
                info = item['info']
                data = item['data']
                for i in range(len(data)):
                    for j in range(len(data)):
                        pred_week = data[j][0]
                        diff_week = pred_week - data[i][0]
                        temp_x = np.concatenate(
                            [info, data[i], np.array([pred_week, np.abs(diff_week), np.square(diff_week), np.tanh(diff_week)])]
                        )
                        temp_y = data[j][1]
                        if features is not None:
                            temp_x = np.concatenate([temp_x, features[k]])

                        self.x.append(temp_x)
                        self.y.append(temp_y)

        elif tag == 'val':
            for k, item in enumerate(bundle):
                info = item['info']
                data = item['data']
                for i in range(min(5, len(data))):
                    for j in [-1, -2, -3]:
                        pred_week = data[j][0]
                        diff_week = pred_week - data[i][0]
                        temp_x = np.concatenate(
                            [info, data[i], np.array([pred_week, np.abs(diff_week), np.square(diff_week), np.tanh(diff_week)])]
                        )
                        temp_y = data[j][1]
                        if features is not None:
                            temp_x = np.concatenate([temp_x, features[k]])

                        self.x.append(temp_x)
                        self.y.append(temp_y)

        elif tag == 'test':
            self.y = None
            for k, item in enumerate(bundle):
                info = item['info']
                data = item['data']
                for i in range(len(data)):
                    for j in range(-12, 134, 1):
                        pred_week = codec_w.encode(j)
                        diff_week = pred_week - data[i][0]
                        temp_x = np.concatenate(
                            [info, data[i], np.array([pred_week, np.abs(diff_week), np.square(diff_week), np.tanh(diff_week)])]
                        )
                        if features is not None:
                            temp_x = np.concatenate([temp_x, features[k]])
                        self.x.append(temp_x)

        else:
            raise KeyError

    def __getitem__(self, idx):
        if self.y is not None:
            return self.x[idx], self.y[idx]
        else:
            return self.x[idx]

    def __len__(self):
        return len(self.x)


class LsmModel:
    def __init__(self):
        self.a = None
        self.b = None

    def save_model(self, output_file):
        temp = np.stack([self.a, self.b], axis=0)
        np.save(output_file, temp)

    def load_model(self, model_file):
        temp = np.load(model_file)
        self.a = temp[0]
        self.b = temp[1]

    def forward(self, x):
        if self.a is None:
            raise RuntimeError

        x = x.detach().cpu().numpy()
        x = np.concatenate([x, np.ones((len(x), 1))], axis=1)
        y = np.matmul(x, self.a)

        return y

    def predict(self, data_set):
        if self.a is None:
            raise RuntimeError

        x = np.array([data for data in data_set], np.float32)
        x = np.concatenate([x, np.ones((len(x), 1))], axis=1)

        y_pred = np.matmul(x, self.a)
        s_pred = np.matmul(x, self.b)

        y_pred = codec_f.decode(y_pred)
        s_pred = codec_f.decode(s_pred, scale_only=True)

        return y_pred, s_pred

    def norm(self, y_pred, y_true):
        return np.mean(np.abs(y_pred - y_true))

    def laplace_log_likelihood(self, y_pred, y_true, sigma):
        delta = np.abs(y_pred - y_true)
        delta = np.minimum(delta, 1000)
        sigma = np.maximum(sigma, 70)

        metric = -1. * (math.sqrt(2) * delta / sigma + np.log(math.sqrt(2) * sigma))
        metric = np.mean(metric)

        return metric


    def train(self, data_set):
        x = np.array([data[0] for data in data_set], np.float32)
        y = np.array([data[1] for data in data_set], np.float32)

        x = np.concatenate([x, np.ones((len(x), 1))], axis=1)
        self.a = np.linalg.lstsq(x, y, rcond=None)[0]

        y_pred = np.matmul(x, self.a)
        s = np.abs(y_pred - y) * math.sqrt(2)

        self.b = np.linalg.lstsq(x, s, rcond=None)[0]
        s_pred = np.matmul(x, self.b)

        y_true = codec_f.decode(y)
        y_pred = codec_f.decode(y_pred)
        s_pred = codec_f.decode(s_pred, scale_only=True)

        norm = self.norm(y_pred, y_true)
        metric = self.laplace_log_likelihood(y_pred, y_true, s_pred)

        return norm, metric

    def val(self, data_set):
        x = np.array([data[0] for data in data_set], np.float32)
        y = np.array([data[1] for data in data_set], np.float32)

        x = np.concatenate([x, np.ones((len(x), 1))], axis=1)

        y_pred = np.matmul(x, self.a)
        s_pred = np.matmul(x, self.b)

        y_true = codec_f.decode(y)
        y_pred = codec_f.decode(y_pred)
        s_pred = codec_f.decode(s_pred, scale_only=True)

        norm = self.norm(y_pred, y_true)
        metric = self.laplace_log_likelihood(y_pred, y_true, s_pred)

        return norm, metric

    def fit(self, train_set, val_set=None):
        if val_set is not None:
            print('training on {} samples, validating on {} samples\n'.format(len(train_set), len(val_set)))
        else:
            print('training on {} samples\n'.format(len(train_set)))

        start_time = time.time()
        norm, metric = self.train(train_set)

        if val_set is not None:
            val_norm, val_metric = self.val(val_set)
            print('[{:2.4f} s] training [norm: {:1.5f}, metric: {:1.5f}], validation [norm: {:1.5f}, metric: {:1.5f}]'.format(
                time.time() - start_time, norm, metric, val_norm, val_metric
            ))
        else:
            print('[{:2.4f} s] training [norm: {:1.5f}, metric: {:1.5f}]'.format(
                time.time() - start_time, norm, metric
            ))


# predict.py
def write_csv(patients_id, y, c, output_file):
    make_dir(output_file)
    with open(output_file, 'w') as f:
        f.write('Patient_Week,FVC,Confidence\n')

        for w in range(146):
            for i, p in enumerate(patients_id):
                f.write('{}_{},{},{}\n'.format(p, w - 12, y[i * 146 + w], c[i * 146 + w]))


def predict_lsm(test_csv, model_file, output_file):
    with open(test_csv) as f:
        content = f.read().splitlines()[1:]
        content = [e.split(',') for e in content]
        patients_id = [e[0] for e in content]

    test_bundle = process_data(TEST_CSV, None, interpolation=False, add_noise=False)
    test_set = LsmDataset(test_bundle, tag='test')

    model = LsmModel()
    model.load_model(model_file)
    temp = model.predict(test_set)
    y = temp[0]
    c = temp[1]

    write_csv(patients_id, y, c, output_file)


if __name__ == '__main__':
    predict_lsm(TEST_CSV, MODEL_FILE, SUBMIT_CSV)
