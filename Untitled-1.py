
import numpy as np
#dataset 数据集
#作用：存储数据集信息，self.xxx
#      获取数据集长度：__len__
#      获取数据集某特定条目内容：__getitem__

class ImageDataset:
    def __init__(self, raw_data):
        self.raw_data = raw_data
    
    def __len__(self):
        return len(self.raw_data)
    
    def __getitem__(self, index):
        image, label = self.raw_data[index]#返回一个元组
        return image, label
        
       #dataloader 数据加载器
#作用：从数据集中加载数据，并拼接为一个数据batch
#       实现迭代器，让使用者获取数据集的内容

class DataLoader:
    def __init__(self, dataset, batch_size):
        self.dataset = dataset
        self.batch_size = batch_size
       
        
    def __iter__(self):
        #正常序列 并重置指针
        self.indexs = np.arange(len(self.dataset))
        self.cursor = 0
        #打乱序列
        np.random.shuffle(self.indexs)
        
        return self
    
    def __next__(self):
            # 预期在这里返回一个batch数据，一个batch是随机抓取的
            # 抓batch个数据前，先抓batch个index
        begin = self.cursor
        end = self.cursor + self.batch_size
        if end > len(self.dataset):
            raise StopIteration
            
        self.cursor = end    
        select_index = self.indexs[begin:end]
        batch_data = []
        for index in select_index:
            batch_data.append(self.dataset[index])
        return batch_data
        
images = [[f"image{i}", i] for i in range(10)]

dataset = ImageDataset(images)
loader = DataLoader(dataset, batch_size=5)
# for index, batch_data in enumerate(loader):#enumerate 表明的是打包函数，将标签和标签所对应的值一同返回
#     print(batch_data)
for batch_data in loader:
    print(batch_data)

