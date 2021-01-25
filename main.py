from model import *
import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import TensorDataset, DataLoader
from torch.nn import functional as F
from torchvision import models
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import argparse

# tensorboard用の処理
writer = SummaryWriter()

BATCH_SIZE=100
EPOCHS=200

# モデルの出力名の設定
parser=argparse.ArgumentParser()
parser.add_argument("export")
args=parser.parse_args()

def train(loader):
    D.train()
    G.train()
    # optimizerの設定
    D_optimizer = torch.optim.Adam(D.parameters(), lr=DiscriminatorLR, betas=(0.9, 0.999))
    G_optimizer = torch.optim.Adam(G.parameters(), lr=GeneratorLR, betas=(0.9, 0.999))

    realLabel = torch.ones(BATCH_SIZE, 1).cuda()
    fakeLabel = torch.zeros(BATCH_SIZE, 1).cuda()
    BCE = torch.nn.BCELoss()
    VggLoss = VGGLoss()

    for batch_idx, (X, Y) in tqdm(enumerate(loader),total=400):
        if X.shape[0] < BATCH_SIZE:
            break

        X = X.cuda()
        Y = Y.cuda()

        # Gのパラメータを更新
        D.eval()
        fakeFrame = G(X)
        DFake = D(fakeFrame)
        G.zero_grad()
        G_label_loss= BCE(DFake, realLabel)
        G_loss=VGGLoss()
        MSELoss=nn.MSELoss()
        G_loss=G_loss(fakeFrame,Y)*1e-3
        G_loss += (G_label_loss+MSELoss(fakeFrame,Y))
        G_loss.backward()
        G_optimizer.step()

        # Gのパラメータを更新
        D.train()
        D.zero_grad()
        DReal = D(Y)
        DFake = D(fakeFrame.detach())
        loss= nn.BCELoss()
        D_loss = loss(DFake, fakeLabel) + loss(DReal, realLabel)
        D_loss.backward()
        D_optimizer.step()

        # tensorboard用に損失を保存
        writer.add_scalar("GLoss", G_loss, epoch)
        writer.add_scalar("DLoss", D_loss, epoch)
    # modelの保存
    # torch.save(D.state_dict(),'./models/'+args.export+'_D_'+str(epoch)+'.pt')
    # torch.save(G.state_dict(),'./models/'+args.export+'_G_'+str(epoch)+'.pt')
    writer.flush()

if __name__ == '__main__':
    G = Generator()
    D = Discriminator()
    G = G.cuda()
    D = D.cuda()
    G.load_state_dict(torch.load('./models/VGG_G_199.pt'))
    D.load_state_dict(torch.load('./models/VGG_D_199.pt'))

    # LRの設定
    GeneratorLR = 0.001
    DiscriminatorLR = 0.0005

    # data load
    train_x = (np.load('fake_train_50k.npy')/127.5)-1
    train_y = (np.load('STL10_train_50k.npy')/127.5)-1
    test_x = (np.load('fake_test_50k.npy')/127.5)-1
    test_y = (np.load('STL10_test_50k.npy')/127.5)-1
    print("data loaded")

    # tensor datasetの作成
    tensor_tr_x, tensor_tr_y = torch.tensor(train_x, dtype=torch.float), torch.tensor(train_y, dtype=torch.float)
    DS_TR = TensorDataset(tensor_tr_x.cuda(), tensor_tr_y.cuda())
    tensor_te_x, tensor_te_y = torch.tensor(test_x, dtype=torch.float), torch.tensor(test_y, dtype=torch.float)
    DS_TE = TensorDataset(tensor_te_x.cuda(), tensor_te_y.cuda())
    
    # data loaderの作成
    train_loader = DataLoader(DS_TR ,batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(DS_TE ,batch_size=BATCH_SIZE, shuffle=True)
    
    # 学習
    for epoch in range(EPOCHS):
        print('{}/{}'.format(epoch+1,EPOCHS))
        train(train_loader)