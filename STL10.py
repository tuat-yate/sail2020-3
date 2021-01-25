from torchvision.datasets import STL10
from torchvision import transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

# データセットのダウンロード
STL10_dataset=STL10("STL10",split='unlabeled', download=True, transform=transforms.ToTensor())

# データをシャッフル
np.random.shuffle(STL10_dataset.data)
# 40000画像をtrain，10000画像をtest
STL10_train = STL10_dataset.data[:40000]
STL10_test = STL10_dataset.data[40000:50000]

# bicubic補間用のndarrayを生成
fake_train=np.zeros((STL10_train.shape[0],3,24,24))
for i in range(STL10_train.shape[0]):
    print(i,'/40000')
    # PILで扱える形に変換
    img=Image.fromarray(STL10_train[i].transpose(1,2,0))
    # bicubic補間
    img=img.resize((int(img.width/4), int(img.height/4)), Image.BICUBIC)
    # ch*24*24の形に直して保存
    fake_train[i]=np.asarray(img).transpose(2,0,1)
fake_train=fake_train.astype('uint8')

np.save('fake_train_50k',fake_train)
np.save('STL10_train_50k',STL10_train.data)
del(fake_train)
del(STL10_train)

# testデータも同様に生成
fake_test=np.zeros((STL10_test.shape[0],3,24,24))
for i in range(STL10_test.shape[0]):
    print(i,'/10000')
    img=Image.fromarray(STL10_test[i].transpose(1,2,0))
    img=img.resize((int(img.width/4), int(img.height/4)), Image.BICUBIC)
    fake_test[i]=np.asarray(img).transpose(2,0,1)
fake_test=fake_test.astype('uint8')

np.save('fake_test_50k',fake_test)
np.save('STL10_test_50k',STL10_test.data)