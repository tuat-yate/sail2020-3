from torchvision.datasets import STL10
from torchvision import transforms
from torch.utils.data import DataLoader,TensorDataset
import matplotlib.pyplot as plt
import torch.nn.functional as F
from PIL import Image
import numpy as np
import torch
from torch import nn,optim
from torchvision import models
BATCH_SIZE=100

STL10_train = STL10("STL10", split='train', download=True, transform=transforms.ToTensor())
STL10_test = STL10("STL10", split='test', download=True, transform=transforms.ToTensor())

fake_train=np.zeros((STL10_train.data.shape[0],3,24,24))
for i in range(STL10_train.data.shape[0]):
    img=Image.fromarray(STL10_train.data[i].transpose(1,2,0))
    img=img.resize((int(img.width/4), int(img.height/4)), Image.BICUBIC)
    fake_train[i]=np.asarray(img).transpose(2,0,1)
fake_train=fake_train.astype('uint8')

fake_test=np.zeros((STL10_test.data.shape[0],3,24,24))
for i in range(STL10_test.data.shape[0]):
    img=Image.fromarray(STL10_test.data[i].transpose(1,2,0))
    img=img.resize((int(img.width/4), int(img.height/4)), Image.BICUBIC)
    fake_test[i]=np.asarray(img).transpose(2,0,1)
fake_test=fake_test.astype('uint8')

np.save('fake_train',fake_train)
np.save('STL10_train',STL10_train.data)
np.save('fake_test',fake_test)
np.save('STL10_test',STL10_test.data)

#train_x:24*24
#train_y:96*96
train_x=torch.tensor(fake_train,dtype=torch.float)
train_y=torch.tensor(STL10_train.data,dtype=torch.float)
test_x=torch.tensor(fake_test,dtype=torch.float)
test_y=torch.tensor(STL10_test.data,dtype=torch.float)
    
train_loader = DataLoader(TensorDataset(train_x,train_y), batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(TensorDataset(test_x,test_y), batch_size=BATCH_SIZE, shuffle=True)