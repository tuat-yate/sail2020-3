# SAIL2020-3
![SRGAN](https://github.com/tuat-yate/sail2020-3/blob/main/out.jpg)
This repository has the code created in "Frontiers in Electrical Engineering and Computer Science: Experiments III"

These codes are implementations of SRGAN

## codes

- `main.py`

  Run the model using `model.py`

- `model.py`

  Generator & Discriminator & each loss function

- `STL10.py`

  Make own dataset using STL10 from `torchvision`

- `eval.py`

  Evaluate the model

## Requirement
```
Python == 3.8.5
torch == 1.7.1+cu110
CUDA version == 11.2
Driver version 460.32.03
torchvison
matplotlib
tqdm
PIL
skimage
pandas
```


## Usage

```
git clone https://github.com/tuat-yate/sail2020-3
cd sail2020-3
python STL10.py
python main.py VGG
python eval.py
```



