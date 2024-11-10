import os
import torch.nn as nn
from torch.nn import Linear
import numpy as np
import struct
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torch.autograd import Variable
from PIL import Image
data_path = './data/MNIST/raw'

def load_mnist(path, kind = 'train'):
    '''
    :params path: 说明要解压的路径
    :params kind: 说明是train还是test

    :return images: 是一个 (样本数 * 特征数) 的数组
    :return labels: 类标签
    '''
    # 标签地址
    labels_path = os.path.join(
        path, 
        '%s-labels-idx1-ubyte' % kind
    )
    # 图片地址
    images_path = os.path.join(
        path,
        '%s-images-idx3-ubyte' % kind
    )

    with open(labels_path, 'rb') as lbpath:
        # 以二进制形式读取标签地址
        magic, n = struct.unpack(
            '>II',
            lbpath.read(8)
        )
        labels = np.fromfile(
            lbpath,
            dtype=np.uint8
        )
    
    with open(images_path, 'rb') as imgpath:
        magic, num, rows, cols = struct.unpack(
            '>IIII',
            imgpath.read(16)
        )
        images = np.fromfile(
            imgpath,
            dtype=np.uint8
        ).reshape( len(labels), 784)  # 784 = 28 * 28

    return images, labels

X_train, y_train = load_mnist(data_path, kind='train')
X_test, y_test = load_mnist(data_path, kind='t10k')
fig, ax = plt.subplots(  
    nrows = 2,
    ncols = 5,
    sharex=True,
    sharey=True
)  # 创建2行5列的画布
ax = ax.flatten()

for i in range(10):
    # train数据集中，label等于0-9的第一个图片
    img = X_train[y_train == i][0].reshape(28, 28)  
    ax[i].imshow(img, cmap='Greys', interpolation='nearest') 
ax[0].set_xticks([]) # 舍去坐标信息
ax[0].set_yticks([])
# plt.tight_layout()
plt.show()
fig, ax = plt.subplots( # 重置画布
    nrows=5,
    ncols=5,
    sharex=True,
    sharey=True, )

ax = ax.flatten()
for i in range(25):
    img = X_train[y_train == 5][i].reshape(28, 28)
    ax[i].imshow(img, cmap='Greys', interpolation='nearest')

ax[0].set_xticks([])
ax[0].set_yticks([])
plt.tight_layout()
plt.show()
train_transform = transforms.Compose([
    transforms.Resize((28, 28)),
    # transforms.RandomCrop(),
    transforms.ToTensor(),
    transforms.Normalize([0.5],   
                         [0.5],) # 由于创建的是1通道的，所以标准差和平均值都是一维
])

# 自定义数据集类
class myData(Dataset):
    def __init__(self, path, transform, kind):
    	super(myData, self).__init__()
        self.path = path
        self.transform = transform
        self.data_info, self.labels = load_mnist(path, kind)

    def __getitem__(self, index):
        img = self.data_info[index]
        img = Image.fromarray(img) # transforms功能只能作用于img格式
        label = torch.tensor(self.labels[index], dtype=torch.long)
        if self.transform is not None:
            img = self.transform(img)
        return img, label
    
    def __len__(self):
        return len(self.data_info)
# 实例化数据集
trainset = myData(data_path,train_transform ,'train')
testset = myData(data_path,train_transform,'t10k')

train_loader = DataLoader(
    dataset=trainset,
    batch_size=100,
    shuffle=True
)

test_loader = DataLoader(
    dataset=testset,
    batch_size=100,
    shuffle=False
)
