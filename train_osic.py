import time, os, sys, pickle, math, random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt

from nets import *
from utils import *
from train_lsm import LsmModel
from data_process import process_data


NUM_WORKERS = 6

def fix_random_seed(random_seed):
    torch.manual_seed(random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(random_seed)
        torch.cuda.manual_seed_all(random_seed)
    np.random.seed(random_seed)
    random.seed(random_seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


class OsicDataset(Dataset):
    def __init__(self, bundle, transform, tag='train', sample_num=20):
        self.x = []
        self.y = []
        # self.images = [item['image'] for item in arr]
        self.bundle = bundle
        self.idx_to_image_id = []
        self.transofrm = transform
        self.tag = tag
        self.sample_num = sample_num

        if tag == 'train':
            self.length = sample_num * len(bundle)

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

                        self.x.append(temp_x)
                        self.y.append(temp_y)
                        self.idx_to_image_id.append(k)
            self.length = len(self.x)

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

                        self.x.append(temp_x)
                        self.idx_to_image_id.append(k)
            self.length = len(self.x)

        else:
            raise KeyError

    def __getitem__(self, idx):
        if self.tag == 'train':
            item = self.bundle[idx // self.sample_num]
            info = item['info']
            data = item['data']
            images = item['images']

            i = random.randint(0, len(data) - 1)
            j = random.randint(0, len(data) - 1)

            pred_week = data[j][0]
            diff_week = pred_week - data[i][0]
            x = np.concatenate(
                [info, data[i], np.array([pred_week, np.abs(diff_week), np.square(diff_week), np.tanh(diff_week)])]
            )
            y = data[j][1]



            # y += codec_f.encode(70, scale_only=True) * np.random.normal()

            image_id = random.randint(0, len(images) - 1) if images is not None else None
        else:
            x = self.x[idx]
            y = self.y[idx] if self.y is not None else None
            images = self.bundle[self.idx_to_image_id[idx]]['images']

            image_id = len(images) // 2 if images is not None else None

        image = self.transofrm(images[image_id]) if images is not None else None

        # return image, x, y
        return x, y

    def __len__(self):
        return self.length


class OsicModel:
    def __init__(self, name='_', lsm_model=None, net=NetOsic(), learning_rate=0.001, step_size=20, gamma=0.7):
        self.name = name
        self.epoch = 0
        self.losses = []
        self.lsm_model = lsm_model

        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        print('DEVICE: {}'.format(self.device))

        self.net = net.to(self.device)
        # self.optimizer = optim.Adam(self.net.parameters(), lr=learning_rate)
        self.optimizer = optim.Adam(self.net.parameters(), lr=learning_rate, weight_decay=1e-2)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=step_size, gamma=gamma)

    def print_model(self):
        print(self.net)
        print('total trainable parameters: {}'.format(sum(p.numel() for p in self.net.parameters() if p.requires_grad)))

    def save_checkpoint(self, output_file, weights_only=False):
        checkpoint = {
            'epoch': self.epoch,
            'losses': self.losses,
            'net_state_dict': self.net.state_dict(),
        }
        if not weights_only:
            checkpoint.update({
                'optimizer_state_dict': self.optimizer.state_dict()
            })
        make_dir(output_file)
        torch.save(checkpoint, output_file)

    def load_checkpoint(self, checkpoint_file, weights_only=False):
        checkpoint = torch.load(checkpoint_file, map_location=self.device)
        self.net.load_state_dict(checkpoint['net_state_dict'])
        if not weights_only:
            self.epoch = checkpoint['epoch'] + 1
            self.scheduler.last_epoch = checkpoint['epoch']
            self.losses = checkpoint['losses']
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    # def predict(self, test_set, batch_size=4):
    #     output = []
    #     self.net.eval()
    #     test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=NUM_WORKERS)
    #     with torch.no_grad():
    #         for _, data in enumerate(test_loader):
    #             # x = data[0].to(self.device, torch.float)
    #             x = data.to(self.device, torch.float)

    #             y_pred = self.net(x)
    #             y_pred = y_pred.detach().squeeze().cpu().numpy()
    #             output.extend(y_pred)

    #     output = np.array(output, np.float32)
    #     return output

    def pinball_loss(self, y_pred, y_true):
        y_true = y_true.unsqueeze(dim=1)

        q = torch.FloatTensor([0.5, 0.2, 0.8]).to(self.device)
        e = y_true - y_pred

        value = torch.max(q * e, (q - 1) * e)
        value = torch.mean(value)
        return value

    def laplace_log_likelihood(self, y_pred, y_true):
        ones = torch.ones(y_pred[:, 0].size()).to(self.device)

        delta = torch.abs(y_pred[:, 0] - y_true)
        delta = torch.min(delta, ones * 1000)

        # sigma = y_pred[:, 2] - y_pred[:, 1]
        sigma = y_pred[:, 1]
        sigma = torch.max(sigma, ones * 70)

        metric = math.sqrt(2) * delta / sigma + torch.log(math.sqrt(2) * sigma)
        metric = torch.mean(metric)
        return metric

    def train_on_epoch(self, loader, alpha=0.8):
        self.net.train()
        loss = 0.
        norm = 0.
        metric = 0.
        for _, data in enumerate(loader):
            self.optimizer.zero_grad()

            # image = data[0].to(self.device, torch.float) if data[0] is not None else None
            image = None
            x = data[0].to(self.device, torch.float)
            y = data[1].to(self.device, torch.float)

            y_pred = self.net(image, x).squeeze()

            # if self.lsm_model is not None:
            #         y_pred = y_pred + torch.FloatTensor(self.lsm_model.forward(x)).to(self.device)

            batch_loss = F.mse_loss(y_pred, y)
            batch_loss.backward()

            self.optimizer.step()
            loss += batch_loss.item()
            norm += codec_f.decode(F.l1_loss(y_pred, y).item(), scale_only=True)

        return loss / len(loader), norm / len(loader), metric / len(loader)

    def val_on_epoch(self, loader, alpha=0.8):
        self.net.eval()
        with torch.no_grad():
            loss = 0.
            norm = 0.
            metric = 0.
            for _, data in enumerate(loader):
                # image = data[0].to(self.device, torch.float) if data[0] is not None else None
                image = None
                x = data[0].to(self.device, torch.float)
                y = data[1].to(self.device, torch.float)

                y_pred = self.net(image, x).squeeze()

                # if self.lsm_model is not None:
                #     y_pred = y_pred + torch.FloatTensor(self.lsm_model.forward(x)).to(self.device)

                batch_loss = F.mse_loss(y_pred, y)

                loss += batch_loss
                norm += codec_f.decode(F.l1_loss(y_pred, y).item(), scale_only=True)

        return loss / len(loader), norm / len(loader), metric / len(loader)

    def fit(self, train_set, val_set=None, epochs=1, batch_size=32, checkpoint=False, save_progress=False, final_model=False):
        def routine():
            print('epoch {:>3} [{:2.2f} s] '.format(self.epoch + 1, time.time() - start_time), end='')

            if validate:
                print('training [loss: {:3.7f}, norm: {:1.5f}, metric: {:1.5f}], validation [loss: {:3.7f}, norm: {:1.5f}, metric: {:1.5f}]'.format(
                    loss, norm, metric, val_loss, val_norm, val_metric
                ))
                if save_progress:
                    self.losses.append((loss, norm, metric, val_loss, val_norm, val_metric))
                if checkpoint and (self.epoch + 1) % checkpoint == 0:
                    folder = './model/{}'.format(self.name)
                    make_dir(folder)
                    if final_model:
                        self.save_checkpoint('{}/e{:02}.pickle'.format(folder, self.epoch + 1), weights_only=True)
                    else:
                        self.save_checkpoint('{}/e{:02}_v{:.4f}.pickle'.format(folder, self.epoch + 1, abs(val_metric)))
            else:
                print('training [loss: {:3.7f}, norm: {:1.5f}, metric: {:1.5f}]'.format(
                    loss, norm, metric
                ))
                if save_progress:
                    self.losses.append((loss, norm, metric))
                if checkpoint and (self.epoch + 1) % checkpoint == 0:
                    folder = './model/{}'.format(self.name)
                    make_dir(folder)
                    if final_model:
                        self.save_checkpoint('{}/e{:02}.pickle'.format(folder, self.epoch + 1), weights_only=True)
                    else:
                        self.save_checkpoint('{}/e{:02}_t{:.4f}.pickle'.format(folder, self.epoch + 1, abs(metric)))


        validate = True if val_set is not None else False
        train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=NUM_WORKERS)

        if validate:
            val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=NUM_WORKERS)
            print('training on {} samples, validating on {} samples\n'.format(len(train_set), len(val_set)))
        else:
            print('training on {} samples\n'.format(len(train_set)))

        while self.epoch < epochs:
            start_time = time.time()
            loss, norm, metric = self.train_on_epoch(train_loader)
            if validate:
                val_loss, val_norm, val_metric = self.val_on_epoch(val_loader)

            routine()
            self.scheduler.step()
            self.epoch += 1


def main():
    bundle = process_data('raw/train.csv', add_noise=False)
    train_bundle, val_bundle = bundle[:160], bundle[160:]

    train_set = OsicDataset(train_bundle, train_transform, tag='train', sample_num=100)
    val_set = OsicDataset(val_bundle, val_transform, tag='val')

    lsm_model = LsmModel()
    lsm_model.load_model('model/lsm_09.npy')

    model = OsicModel('cnn_01', lsm_model=lsm_model, net=NetFc(input_dim=13, input_channel=1, output_dim=1), learning_rate=5e-5)
    model.fit(train_set, val_set, epochs=200, batch_size=32)


if __name__ == '__main__':
    fix_random_seed(42)
    main()
