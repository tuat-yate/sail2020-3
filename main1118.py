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
writer = SummaryWriter()

BATCH_SIZE=100
EPOCHS=128
def train(loader):
    D.train()
    G.train()

    D_optimizer = torch.optim.Adam(D.parameters(), lr=DiscriminatorLR, betas=(0.9, 0.999))
    G_optimizer = torch.optim.Adam(G.parameters(), lr=GeneratorLR, betas=(0.9, 0.999))

    realLabel = torch.ones(BATCH_SIZE, 1).cuda()
    fakeLabel = torch.zeros(BATCH_SIZE, 1).cuda()
    #realLabel = torch.ones(BATCH_SIZE, 1)
    #fakeLabel = torch.zeros(BATCH_SIZE, 1)
    BCE = torch.nn.BCELoss()
    VggLoss = VGGLoss()

    for batch_idx, (X, Y) in tqdm(enumerate(loader),total=50):
        if X.shape[0] < BATCH_SIZE:
            break

        X = X.cuda()
        Y = Y.cuda()

        D.eval()
        fakeFrame = G(X)
        #print(batch_idx,'/100')
        DFake = D(fakeFrame)

        G.zero_grad()
        #print(torch.sigmoid(DFake))
        G_label_loss= BCE(DFake, realLabel)
        writer.add_scalar("G_label_Loss", G_label_loss, epoch)
        #print(fakeFrame.shape)
        #print(Y.shape)
        G_loss = VggLoss(fakeFrame, Y)
        writer.add_scalar("GLoss", G_loss, epoch)
        G_loss += 1e-3 * G_label_loss
        G_loss.backward()
        G_optimizer.step()

        D.train()
        D.zero_grad()
        DReal = D(Y)
        DFake = D(fakeFrame.detach())
        loss= nn.BCEWithLogitsLoss()
        #print(fakeLabel.dtype)
        #print(loss(DFake, fakeLabel))
        D_loss = loss(DFake, fakeLabel) + loss(DReal, realLabel)
        D_loss.backward()
        D_optimizer.step()


        #writer.add_scalar("GLoss", G_loss, epoch)
        writer.add_scalar("DLoss", D_loss, epoch)
        #print("G_loss :", G_loss, " D_loss :", D_loss)
        torch.save(D.state_dict(),'D.pth')
        torch.save(G.state_dict(),'G.pth')
    writer.flush()


def save_imgs(epoch, data,datax2):
    r = 5
    #print('plot...')
    G.eval()
    t_data = data
    genImgs = G(torch.tensor(t_data, dtype=torch.float).cuda())
    genImgs = genImgs.cpu().detach().numpy()
    #print('max',np.max(genImgs))
    #print('min',np.min(genImgs))
    #genImgs = genImgs / 2 + 0.5
    data = np.transpose((data+1)*127.5, (0, 2, 3, 1)).astype(np.uint8)
    genImgs = np.transpose((genImgs+1)*127.5, (0, 2, 3, 1)).astype(np.uint8)
    datax2 = np.transpose((datax2+1)*127.5, (0, 2, 3, 1)).astype(np.uint8)

    fig, axs = plt.subplots(3, r)
    for i in range(r):
        axs[0, i].imshow(data[i, :, :, :])
        axs[0, i].axis('off')
        axs[1, i].imshow(genImgs[i, :, :, :])
        axs[1, i].axis('off')
        axs[2, i].imshow(datax2[i, :, :, :])
        axs[2, i].axis('off')

    fig.savefig("generat_images/gen_%d.png" % epoch)
    plt.close()


if __name__ == '__main__':
    G = Generator()
    D = Discriminator()

    G = G.cuda()
    D = D.cuda()

    GeneratorLR = 0.00025
    DiscriminatorLR = 0.00001

    train_x = (np.load('fake_train.npy')/127.5)-1
    train_y = (np.load('STL10_train.npy')/127.5)-1

    test_x = (np.load('fake_test.npy')/127.5)-1
    test_y = (np.load('STL10_test.npy')/127.5)-1

    tensor_tr_x, tensor_tr_y = torch.tensor(train_x, dtype=torch.float), torch.tensor(train_y, dtype=torch.float)
    DS_TR = TensorDataset(tensor_tr_x.cuda(), tensor_tr_y.cuda())

    tensor_te_x, tensor_te_y = torch.tensor(test_x, dtype=torch.float), torch.tensor(test_y, dtype=torch.float)
    DS_TE = TensorDataset(tensor_te_x.cuda(), tensor_te_y.cuda())
    #DS = TensorDataset(tensor_x, tensor_y)
    train_loader = DataLoader(DS_TR ,batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(DS_TE ,batch_size=BATCH_SIZE, shuffle=True)
    for epoch in range(EPOCHS):
        print('{}/{}'.format(epoch+1,EPOCHS))
        train(train_loader)
        save_imgs(epoch,test_x[:100],test_y[:100])