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





