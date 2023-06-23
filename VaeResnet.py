import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import transforms, utils
from torch.utils.data import Dataset, DataLoader
import torch.utils.data as Data
import torchvision.models as models
import numpy as np
from torchvision.datasets import ImageFolder


# 用上采样加卷积代替了反卷积
class ResizeConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, scale_factor, mode='nearest'):
        super().__init__()
        self.scale_factor = scale_factor
        self.mode = mode
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=1)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=self.scale_factor, mode=self.mode)
        x = self.conv(x)
        return x


class ResNet18Enc(nn.Module):
    def __init__(self, z_dim=32):
        super(ResNet18Enc, self).__init__()
        self.z_dim = z_dim
        self.ResNet18 = models.resnet18(pretrained=True)
        self.num_feature = self.ResNet18.fc.in_features
        self.ResNet18.fc = nn.Linear(self.num_feature, 2 * self.z_dim)

    def forward(self, x):
        x = self.ResNet18(x)
        mu = x[:, :self.z_dim]
        logvar = x[:, self.z_dim:]
        return mu, logvar


class BasicBlockDec(nn.Module):
    def __init__(self, in_planes, stride=1):
        super().__init__()
        planes = int(in_planes / stride)
        self.conv2 = nn.Conv2d(in_planes, in_planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(in_planes)

        if stride == 1:
            self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
            self.bn1 = nn.BatchNorm2d(planes)
            self.shortcut = nn.Sequential()
        else:
            self.conv1 = ResizeConv2d(in_planes, planes, kernel_size=3, scale_factor=stride)
            self.bn1 = nn.BatchNorm2d(planes)
            self.shortcut = nn.Sequential(
                ResizeConv2d(in_planes, planes, kernel_size=3, scale_factor=stride),
                nn.BatchNorm2d(planes)
            )

    def forward(self, x):
        out = torch.relu(self.bn2(self.conv2(x)))
        out = self.bn1(self.conv1(out))
        out += self.shortcut(x)
        out = torch.relu(out)
        return out


class ResNet18Dec(nn.Module):
    def __init__(self, num_Blocks=[2, 2, 2, 2], z_dim=32, nc=3):
        super().__init__()
        self.in_planes = 256
        self.linear = nn.Linear(z_dim, 256)
        self.layer4 = self._make_layer(BasicBlockDec, 128, num_Blocks[3], stride=2)
        self.layer3 = self._make_layer(BasicBlockDec, 64, num_Blocks[2], stride=2)
        self.layer2 = self._make_layer(BasicBlockDec, 32, num_Blocks[1], stride=2)
        self.layer1 = self._make_layer(BasicBlockDec, 32, num_Blocks[0], stride=1)
        self.conv1 = ResizeConv2d(32, nc, kernel_size=3, scale_factor=2)

    def _make_layer(self, BasicBlockDec, planes, num_Blocks, stride):
        strides = [stride] + [1] * (num_Blocks - 1)
        layers = []
        for stride in reversed(strides):
            layers += [BasicBlockDec(self.in_planes, stride)]
        self.in_planes = planes
        return nn.Sequential(*layers)

    def forward(self, z):
        x = self.linear(z)
        x = x.view(z.size(0), 256, 1, 1)
        x = F.interpolate(x, scale_factor=7)
        x = self.layer4(x)
        x = self.layer3(x)
        x = self.layer2(x)
        x = self.layer1(x)
        x = F.interpolate(x, size=(112, 112), mode='bilinear')
        x = torch.sigmoid(self.conv1(x))
        x = x.view(x.size(0), 3, 224, 224)
        return x


class VAE(nn.Module):
    def __init__(self, z_dim):
        super(VAE, self).__init__()
        self.encoder = ResNet18Enc(z_dim=z_dim)
        self.decoder = ResNet18Dec(z_dim=z_dim)

    def forward(self, x):
        mean, logvar = self.encoder(x)
        z = self.reparameterize(mean, logvar)
        x = self.decoder(z)
        return x, mean, logvar, z

    @staticmethod
    def reparameterize(mean, logvar):
        std = torch.exp(logvar / 2)  # in log-space, squareroot is divide by two
        epsilon = torch.randn_like(std)
        return epsilon * std + mean


def loss_func(recon_x, x, mu, logvar):
    BCE = F.binary_cross_entropy(recon_x, x, reduction='sum')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + KLD, BCE, KLD


class VaeModel:
    def __init__(self):
        self.base_progress = 0
        self.output_path = None
        self.progress_signal = None
        self.train_loader = None
        self.vae = None
        # 初始化参数
        self.epoch_num = 100
        self.batch_size = 8
        # vae = VAE(z_dim=256).cuda()
        self.vae = VAE(z_dim=5)
        self.optimizer = optim.Adam(self.vae.parameters(), lr=1e-4)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=30, gamma=0.1)

    def set_train_input(self, inputDataPath):
        print("[VAE] set dataloader input path: {}.".format(inputDataPath))
        # 数据预处理初始化
        data_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0, 0, 0], std=[1, 1, 1])])
        # 数据加载
        dataset = ImageFolder(inputDataPath, transform=data_transform)
        self.train_loader = Data.DataLoader(dataset=dataset, batch_size=self.batch_size, shuffle=True)
        print("[VAE] DataLoader Init Done, input path: {}".format(inputDataPath))

    def set_progress_updater(self, progress_signal, base_progress):
        self.progress_signal = progress_signal
        self.base_progress = base_progress
        print("[VAE] Set progress done, base: {}".format(base_progress))

    def set_output_model_path(self, output_path):
        self.output_path = output_path
        
    # signalObj是外部进度监视对象
    def train(self):
        # 开始训练
        print("[VAE] Model training enter.")
        for epoch in range(0, self.epoch_num):
            l_sum = 0
            self.scheduler.step()
            for x, y in self.train_loader:
                # x = torch.sigmoid(x).cuda()
                # x = x.cuda()
                self.optimizer.zero_grad()
                recon_x, mu, logvar, z = self.vae.forward(x)
                loss, reconst_loss, kl_div = loss_func(recon_x, x, mu, logvar)
                l_sum += loss
                loss.backward()
                self.optimizer.step()
            if self.progress_signal is not None:
                # 训练固定70%的
                progress = int(epoch * 70 / self.epoch_num) + self.base_progress
                self.progress_signal.emit(progress)
        # 输出模型到特定目录
        output_model_pn = self.output_path + "/dsc_vae_resnet18.pth"
        torch.save(self.vae, output_model_pn)
        self.base_progress += 70
        print("[VAE] Training done, save model to {}".format(output_model_pn))

    def get_train_data_features(self):
        print("[VAE] Get training data feature enter.")
        i = 0
        z_feat = []
        with torch.no_grad():
            total = len(self.train_loader)
            for t_img, y in self.train_loader:
                # t_img = Variable(t_img).cuda()
                t_img = Variable(t_img)
                result, mu, logvar, z = self.vae.forward(t_img)
                if i == 0:
                    z_feat = z
                else:
                    z_feat = torch.cat((z_feat, z))
                i += 1
                progress = int(i * 20 / total) + self.base_progress
                # 防止异步未返回数据就开始处理数据
                if progress < 100:
                    self.progress_signal.emit(progress)
        print("[VAE] Feature process done, type: {}.".format(type(z_feat)))
        return z_feat
