import torch
import torch.nn as nn
import torchvision.transforms as transforms


class Encoder(nn.Module):
    def __init__(self, embed_size=1024):
        super(Encoder, self).__init__()

        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, 3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),

            nn.Conv2d(32, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0),

            nn.Conv2d(64, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),

            nn.Conv2d(128, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0),

            nn.Conv2d(128, 256, 3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),

            nn.Conv2d(256, 256, 3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0),

            nn.Conv2d(256, 512, 3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0),

            nn.Conv2d(512, 512, 3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0),

            nn.AdaptiveAvgPool2d((4, 4))
        )

        self.fc = nn.Linear(512 * 4 * 4, embed_size)

    def forward(self, x):
        x = self.cnn(x)
        x = x.view(x.size()[0], -1)
        x = self.fc(x)
        return x


class Net(nn.Module):
    def __init__(self, embed_size=1024, feature_size=161, output_size=146, train_encoder=False):
        super(Net, self).__init__()

        self.encoder = Encoder(embed_size)
        if not train_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False

        self.gru = nn.GRU(1024, 512, num_layers=1, batch_first=True)

        self.fc = nn.Sequential(
            nn.Linear(512 + feature_size, 512),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(256, output_size)
        )

    def forward(self, images, x):
        batch_num = images.size(0)

        images = images.view(-1, images.size(2), images.size(3), images.size(4))
        embeds = self.encoder(images)
        embeds = embeds.view(batch_num, -1, 1024)

        embed = self.gru(embeds, None)[0][:, -1, :]
        embed = torch.cat((embed, x), dim=1)

        x = self.fc(embed)

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


if __name__ == '__main__':
    images = torch.zeros(3, 20, 1, 512, 512)
    x = torch.zeros(3, 3)
    net = Net()
    y = net(images, x)
    print(y.size())
