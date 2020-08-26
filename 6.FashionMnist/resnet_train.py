import torch
import torchvision
import sys
import time
import numpy as np
import matplotlib.pyplot as plt
from torch import nn
import torch.nn.functional as F


class Resnet(nn.Module):
    # 定义网络结构
    def __init__(self, in_channels, out_channels, use_conv1=False, stride=1):
        super(Resnet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=stride)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        if use_conv1:
            self.conv3 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride)
        else:
            self.conv3 = None
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, X):
        Y = F.relu(self.bn1(self.conv1(X)))
        Y = self.bn2(self.conv2(Y))
        if self.conv3:
            X = self.conv3(X)
        return F.relu(Y + X)


class GlobalAvgPool2d(nn.Module):
    # 将池化窗口形状设置成输入的高和宽
    def __init__(self):
        super(GlobalAvgPool2d, self).__init__()

    def forward(self, x):
        return F.avg_pool2d(x, kernel_size=x.size()[2:])


class FlattenLayer(torch.nn.Module):
    def __init__(self):
        super(FlattenLayer, self).__init__()

    def forward(self, x):  # 改为全连接形式，x shape: (batch, *, *, ...)
        return x.view(x.shape[0], -1)


def load_data_fashion_mnist(batch_sizes, root, resize=None):
    trans = []
    if resize:
        # 将28*28的图片resize为96*96
        trans.append(torchvision.transforms.Resize(size=resize))
    trans.append(torchvision.transforms.ToTensor())  # 转为tensor
    # torchvision.transforms.Resize()
    transform = torchvision.transforms.Compose(trans)
    mnist_train = torchvision.datasets.FashionMNIST(root=root, train=True, download=True, transform=transform)
    mnist_test = torchvision.datasets.FashionMNIST(root=root, train=False, download=True, transform=transform)
    if sys.platform.startswith('win'):
        num_workers = 0  # 不用额外的进程来加速读取数据
    else:
        num_workers = 4
    train_iters = torch.utils.data.DataLoader(mnist_train, batch_size=batch_sizes, shuffle=True, num_workers=num_workers)
    test_iters = torch.utils.data.DataLoader(mnist_test, batch_size=batch_sizes, shuffle=False, num_workers=num_workers)

    return train_iters, test_iters


def resnet_block(in_channels, out_channels, num_residuals, first_block=False):
    # 定义残差块
    if first_block:
        assert in_channels == out_channels  # 第一个模块的通道数同输入通道数一致
    blk = []
    for i in range(num_residuals):
        if i == 0 and not first_block:
            blk.append(Resnet(in_channels, out_channels, use_conv1=True, stride=2))
        else:
            blk.append(Resnet(out_channels, out_channels))
    return nn.Sequential(*blk)


def model():
    net = nn.Sequential(
        nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3),
        nn.BatchNorm2d(64),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
    net.add_module("resnet_block1", resnet_block(64, 64, 2, first_block=True))
    net.add_module("resnet_block2", resnet_block(64, 128, 2))
    net.add_module("resnet_block3", resnet_block(128, 256, 2))
    net.add_module("resnet_block4", resnet_block(256, 512, 2))
    net.add_module("global_avg_pool", GlobalAvgPool2d())  # 输出: (Batch, 512, 1, 1)
    net.add_module("fc", nn.Sequential(FlattenLayer(), nn.Linear(512, 10)))
    print(net)  # 输出网络结构
    return net


def evaluate_accuracy(data_iter, net, device):
    acc_sum, n = 0.0, 0
    with torch.no_grad():
        for X, y in data_iter:
            if isinstance(net, torch.nn.Module):
                net.eval()  # 测试模式,关闭dropout
                acc_sum += (net(X.to(device)).argmax(dim=1) == y.to(device)).float().sum().cpu().item()
                net.train()  # 改回训练模式
            n += y.shape[0]
    return acc_sum / n


def train(net, train_iter, test_iter, batch_size, optimizer, device, num_epochs):
    net = net.to(device)
    print("training on ", device)
    loss = torch.nn.CrossEntropyLoss()  # 多个类别,使用交叉熵损失函数
    batch_count = 0
    for epoch in range(num_epochs):
        train_l_sum, train_acc_sum, n, start = 0.0, 0.0, 0, time.time()
        for X, y in train_iter:
            X = X.to(device)
            y = y.to(device)
            y_hat = net(X)
            l = loss(y_hat, y)
            optimizer.zero_grad()
            l.backward()
            optimizer.step()
            train_l_sum += l.cpu().item()
            train_acc_sum += (y_hat.argmax(dim=1) == y).sum().cpu().item()
            n += y.shape[0]
            batch_count += 1
        test_acc = evaluate_accuracy(test_iter, net,device)
        print('epoch %d, loss %.4f, train acc %.3f, test acc %.3f, time %.1f sec'
          % (epoch + 1, train_l_sum / batch_count, train_acc_sum / n, test_acc, time.time() - start))


def imag_show(iters):
    classes = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
    dataiter = iters
    for i in range(85):
        # 显示第85张图片
        images, labels = dataiter.next()
    images = images.numpy()
    idx = 2
    img = np.squeeze(images[idx])
    # display the pixel values in that image
    fig = plt.figure(figsize=(12, 12))
    ax = fig.add_subplot(111)
    ax.imshow(img, cmap='gray')
    width, height = img.shape
    thresh = img.max() / 2.5
    for x in range(width):
        for y in range(height):
            val = round(img[x][y], 2) if img[x][y] != 0 else 0
            ax.annotate(str(val), xy=(y, x),
                        horizontalalignment='center',
                        verticalalignment='center',
                        color='white' if img[x][y] < thresh else 'black')
    plt.show()


def main():
    lr, num_epochs = 0.001, 8
    net = model()
    train_iter, test_iter = load_data_fashion_mnist(batch_size,
                                                    root=root, resize=96)
    # imag_show(train_iter)
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    train(net, train_iter, test_iter, batch_size, optimizer,
              device, num_epochs)
    torch.save(net.state_dict(), './parameter.pkl')


if __name__ == '__main__':
    resize = 96
    root = './fashion_mnist/'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    batch_size = 256
    main()
