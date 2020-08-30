import time, os, sys, pickle, math, random
import numpy as np
import matplotlib.pyplot as plt

from utils import *
from data_process import process_data, fix_random_seed


NUM_WORKERS = 6


class LsmDataset:
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


def read_csv(csv_file):
    with open(csv_file) as f:
        content = f.read().splitlines()[1:]
        content = [e.split(',') for e in content]

    output = []
    for line in content:
        output.append(float(line[1]))

    output = np.array(output, np.float32)
    output = np.expand_dims(output, -1)
    print(output.shape)
    return output


def main():
    # train_bundle = process_data('raw/train.csv', None, interpolation=True, add_noise=True)

    train_bundle = process_data('raw/train.csv', None, interpolation=True, add_noise=True)[:160]
    val_bundle = process_data('raw/train.csv', None, interpolation=False, add_noise=False)[160:]

    # features = np.load('../osic_version_03/label/test_01.npy')
    # features = read_csv('../osic_version_03/label/offset.csv')
    # train_set = LsmDataset(train_bundle, features=features[:160], tag='train')
    # val_set = LsmDataset(val_bundle, features=features[160:], tag='val')

    train_set = LsmDataset(train_bundle, features=None, tag='train')
    val_set = LsmDataset(val_bundle, features=None, tag='val')
    # val_set = None

    model = LsmModel()
    model.fit(train_set, val_set)

    print(model.a.shape)
    model.save_model('model/lsm_15.npy')


if __name__ == '__main__':
    fix_random_seed(42)
    main()