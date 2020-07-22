# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# Input data files are available in the read-only "../input/" directory
# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All"
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

NUM_WORKERS = 4

X_LENGTH = 161
Y_LENGTH = 146
Y_OFFSET = -12

# TEST_CSV = '../input/osic-pulmonary-fibrosis-progression/test.csv'
# TEST_DIR = '../input/osic-pulmonary-fibrosis-progression/test'
# SUBMIT_CSV = 'submission.csv'
# MODEL_FILE = '../input/test-01/e20_v273.8.pickle'

TEST_CSV = 'raw/test.csv'
TEST_DIR = 'raw/test'
SUBMIT_CSV = 'output/notebook_check.csv'
MODEL_FILE = 'model/test_02/e15_v249.4.pickle'


import os, cv2, pickle, time, random, sys
import pydicom
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms


def make_dir(file_path):
    dirname = os.path.dirname(file_path)
    try:
        if not os.path.exists(dirname):
            os.mkdir(dirname)
    except FileNotFoundError:
        pass


# data_process.py
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


def codec_fcv(value, decode=False):
    if type(value) == str:
        value = float(value)

    if decode:
        return value * 4000.
    else:
        return value / 4000.


def codec_percent(value, decode=False):
    if type(value) == str:
        value = float(value)

    if decode:
        return value * 100.
    else:
        return value / 100.


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
            np.array([codec_fcv(line[2]), codec_percent(line[3])], np.float32),
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
                    cache_y[j] = codec_fcv(user_filter[p][2])

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


# nets.py
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

    def forward(self, x):
        return x


class NetSimple(nn.Module):
    def __init__(self, feature_size=161, output_size=146):
        super(NetSimple, self).__init__()

        self.fc = nn.Sequential(
            nn.Linear(feature_size, 512),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(256, output_size)
        )

    def forward(self, images, x):
        x = self.fc(x)

        return x


train_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ToTensor(),
    transforms.RandomErasing()
])

val_transform = transforms.Compose([
    transforms.ToTensor()
])


# train_osic.py
class ImageDataset(Dataset):
    def __init__(self, images, images_id, x, y, transform):
        self.images = np.expand_dims(images, axis=-1)
        self.images_id = images_id
        self.x = x
        self.y = y
        self.transform = transform

    def __getitem__(self, idx):
        if self.images_id is not None:
            image_id = self.images_id[idx]
        else:
            image_id = idx

        temp = torch.zeros(self.images[image_id].shape).permute(0, 3, 1, 2)
        for i in range(len(self.images[image_id])):
            temp[i, :, :, :] = self.transform(self.images[image_id][i])

        if self.y is not None:
            return temp, self.x[idx], self.y[idx]
        else:
            return temp, self.x[idx]

    def __len__(self):
        return len(self.x)


class OsicModel:
    def __init__(self, name='_', net=Net(), learning_rate=0.001, step_size=20, gamma=0.7):
        self.name = name
        self.epoch = 0
        self.losses = []

        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        print('DEVICE: {}'.format(self.device))

        self.net = net.to(self.device)
        self.optimizer = optim.Adam(self.net.parameters(), lr=learning_rate)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=step_size, gamma=gamma)

    def load_checkpoint(self, checkpoint_file, weights_only=False):
        checkpoint = torch.load(checkpoint_file, map_location=self.device)
        self.net.load_state_dict(checkpoint['net_state_dict'])
        if not weights_only:
            self.epoch = checkpoint['epoch'] + 1
            self.scheduler.last_epoch = checkpoint['epoch']
            self.losses = checkpoint['losses']
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    def predict(self, test_set, batch_size=4):
        output = []
        self.net.eval()
        test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=NUM_WORKERS)
        with torch.no_grad():
            for _, data in enumerate(test_loader):
                images = data[0].to(self.device)
                x = data[1].to(self.device, torch.float)

                y_pred = self.net(images, x)
                y_pred = y_pred.detach().cpu().numpy()
                output.extend(y_pred)

        output = np.array(output, np.float32)
        return output


# predict.py
def write_csv(patients_id, y, c, output_file):
    make_dir(output_file)
    with open(output_file, 'w') as f:
        f.write('Patient_Week,FVC,Confidence\n')

        for w in range(146):
            for i, p in enumerate(patients_id):
                f.write('{}_{},{},{}\n'.format(p, w - 12, y[i][w], c[i][w]))


def predict(test_csv, model_file, output_file):
    with open(test_csv) as f:
        content = f.read().splitlines()[1:]
        content = [e.split(',') for e in content]

    images, images_id, x = process_data(TEST_CSV, TEST_DIR, limit_num=1, return_y=False)

    test_set = ImageDataset(images, images_id, x, None, val_transform)

    model = OsicModel(net=NetSimple())
    model.load_checkpoint(model_file)

    y = model.predict(test_set, batch_size=16)
    y = codec_fcv(y, decode=True).astype(np.int16)

    patients_id = [e[0] for e in content]
    c = np.ones((len(content), 146), np.int16) * 353

    write_csv(patients_id, y, c, output_file)


if __name__ == '__main__':
    predict(TEST_CSV, MODEL_FILE, SUBMIT_CSV)
