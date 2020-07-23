import time, random, os, sys, pickle
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

from nets import *
from utils import *
# from data_process import codec_fcv, codec_percent
# from data_process import codec_f, codec_p

NUM_WORKERS = 4


def make_dir(file_path):
    dirname = os.path.dirname(file_path)
    try:
        if not os.path.exists(dirname):
            os.mkdir(dirname)
    except FileNotFoundError:
        pass


class ImageDataset(Dataset):
    def __init__(self, images, images_id, x, y, transform, add_noise=False):
        self.images = np.expand_dims(images, axis=-1)
        self.images_id = images_id
        self.x = x
        self.y = y
        self.transform = transform
        self.add_noise = add_noise

    def __getitem__(self, idx):
        if self.images_id is not None:
            image_id = self.images_id[idx]
        else:
            image_id = idx

        images = torch.zeros(self.images[image_id].shape).permute(0, 3, 1, 2)
        for i in range(len(self.images[image_id])):
            images[i, :, :, :] = self.transform(self.images[image_id][i])

        if self.y is not None:
            if self.add_noise:
                a = codec_f.encode(500, scale_only=True)
                b = codec_f.encode(100, scale_only=True)

                noise_a = np.random.normal() * a
                noise_b = np.random.normal(size=self.y[0].shape) * b

                x = self.x[idx]
                x[0] += noise_a
                y = self.y[idx] + noise_a + noise_b
            else:
                x = self.x[idx]
                y = self.y[idx]

            return images, x, y
        else:
            return images, self.x[idx]

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

    def train_on_epoch(self, loader):
        self.net.train()
        loss = 0.
        norm = 0.
        for _, data in enumerate(loader):
            self.optimizer.zero_grad()

            images = data[0].to(self.device)
            x = data[1].to(self.device, torch.float)
            y = data[2].to(self.device, torch.float)

            y_pred = self.net(images, x)

            batch_loss = F.mse_loss(y_pred, y)
            batch_loss.backward()

            self.optimizer.step()
            loss += batch_loss.item()
            norm += F.l1_loss(y_pred, y).item()

        return loss / len(loader), norm / len(loader)

    def val_on_epoch(self, loader):
        self.net.eval()
        with torch.no_grad():
            loss = 0.
            norm = 0.
            for _, data in enumerate(loader):
                images = data[0].to(self.device)
                x = data[1].to(self.device, torch.float)
                y = data[2].to(self.device, torch.float)

                y_pred = self.net(images, x)

                loss += F.mse_loss(y_pred, y).item()
                norm += F.l1_loss(y_pred, y).item()

        return loss / len(loader), norm / len(loader)

    def fit(self, train_set, val_set=None, epochs=1, batch_size=32, checkpoint=False, save_progress=False, random_seed=None, final_model=False):
        def routine():
            print('epoch {:>3} [{:2.2f} s] '.format(self.epoch + 1, time.time() - start_time), end='')

            if validate:
                print('training [loss: {:3.7f}, norm: {:1.5f}], validation [loss: {:3.7f}, norm: {:1.5f}]'.format(
                    loss, codec_f.decode(norm, True), val_loss, codec_f.decode(val_norm, True)
                ))
                if save_progress:
                    self.losses.append((loss, norm, val_loss, val_norm))
                if checkpoint and (self.epoch + 1) % checkpoint == 0:
                    folder = './model/{}'.format(self.name)
                    make_dir(folder)
                    if final_model:
                        self.save_checkpoint('{}/e{:02}.pickle'.format(folder, self.epoch + 1), weights_only=True)
                    else:
                        self.save_checkpoint('{}/e{:02}_v{:.1f}.pickle'.format(folder, self.epoch + 1, codec_f.decode(val_norm, True)))
            else:
                print('training [loss: {:3.7f}, norm: {:1.5f}]'.format(
                    loss, codec_f.decode(norm, True)
                ))
                if save_progress:
                    self.losses.append((loss, norm))
                if checkpoint and (self.epoch + 1) % checkpoint == 0:
                    folder = './model/{}'.format(self.name)
                    make_dir(folder)
                    if final_model:
                        self.save_checkpoint('{}/e{:02}.pickle'.format(folder, self.epoch + 1), weights_only=True)
                    else:
                        self.save_checkpoint('{}/e{:02}_t{:.1f}.pickle'.format(folder, self.epoch + 1, codec_f.decode(norm, True)))


        validate = True if val_set is not None else False
        if random_seed is not None:
            torch.manual_seed(random_seed)
            if str(self.device) == 'cuda:0':
                torch.cuda.manual_seed_all(random_seed)

        train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=NUM_WORKERS)
        if validate:
            val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=NUM_WORKERS)
            print('training on {} samples, validating on {} samples\n'.format(len(train_set), len(val_set)))
        else:
            print('training on {} samples\n'.format(len(train_set)))

        while self.epoch < epochs:
            start_time = time.time()
            loss, norm = self.train_on_epoch(train_loader)
            if validate:
                val_loss, val_norm = self.val_on_epoch(val_loader)

            routine()
            self.scheduler.step()
            self.epoch += 1


def main():
    with open('input/train_new.pickle', 'rb') as f:
        images, images_id, x, y = pickle.load(f)

    k = 1200
    val_set = ImageDataset(images, images_id[k:], x[k:], y[k:], val_transform)
    train_set = ImageDataset(images, images_id[:k], x[:k], y[:k], train_transform, add_noise=True)

    # model = OsicModel('_', net=NetSimple(), learning_rate=5e-5, gamma=0.1)
    # model.fit(train_set, val_set, epochs=100, batch_size=8, checkpoint=5)

    model = OsicModel('test_03', net=NetSimple(), learning_rate=5e-5, gamma=0.5)
    model.fit(train_set, val_set, epochs=100, batch_size=8, checkpoint=5)

if __name__ == '__main__':
    main()
