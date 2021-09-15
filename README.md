High-Resolution-MoonNet
===

## Introduction
This is a Keras implementation of [Lunar Features Detection for Energy Discovery via Deep Learning](https://doi.org/10.1016/j.apenergy.2021.117085).

The main network structure is listed as below:
### Model
<div align="center">
<img src="example/Framework.jpg", width="800">
</div>


### Results

<div align="center">
<img src="example/result.jpg", width="800">
</div>

### Data Preparation

For MoonDEM dataset, please download via the folowing link:

```
python data/deepmoon/get_hdf5_data.py
```

For Crack500 and our proposed Assembled dataset with both craters and rilles, you can download via [Google Drive](https://drive.google.com/drive/folders/1PHobsjrkWV6-qDjjNKKuPa85zxK4UNvW?usp=sharing).

After downloading the dataset, please make sure to put all the folders in the the directory:`` data `` and make them looks like this: 


```
├── data
│   ├── assembled_dataset
│   │   ├── assembled_dataset.hdf5
│   │   ├── assembled_train.hdf5
│   │   └── assembled_val.hdf5
│   ├── deepmoon
│   │   ├── dev_craters.hdf5
│   │   ├── dev_images.hdf5
│   │   ├── test_craters.hdf5
│   │   ├── test_images.hdf5
│   │   ├── train_craters.hdf5
│   │   ├── train_images.hdf5
│   │   └── Using\ Zenodo\ Data.ipynb
│   └── surfacecrack
│       └── surface_crack.hdf5
```


## Environment
The code is developed using python 3.6 on Ubuntu 16.04. NVIDIA GPUs are needed. The code is developed and tested using 4 NVIDIA P100 GPU cards. Other platforms or GPU cards are not fully tested.

## Training
**Moon DEM Dataset**
```
python train.py experiment/DeepMoon/High-resolution-net.json
```
**Surface Crack Dataset**
```
python train.py experiment/SurfaceCrack/High-resolution-net.json
```
**Assembled Dataset**
```
python train.py experiment/Assembled/Crack-High-resolution-net.json
python train.py experiment/Assembled/Crater-High-resolution-net.json
```


## Citation

If you use our code or models in your research, please cite with:

```
@article{Chen_2021,	doi = {10.1016/j.apenergy.2021.117085},	
url = {https://doi.org/10.1016%2Fj.apenergy.2021.117085},	
year = 2021,	
month = {aug},	
publisher = {Elsevier {BV}},	volume = {296},	pages = {117085},	
author = {Siyuan Chen and Yu Li and Tao Zhang and Xingyu Zhu and Shuyu Sun and Xin Gao},
title = {Lunar features detection for energy discovery via deep learning},
journal = {Applied Energy}} 
```






