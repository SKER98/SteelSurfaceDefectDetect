# Steel Surface Defect Detection
This is the implementation of the paper "An end‐to‐end steel surface defect detection approach via swin transformer"

## Install the environment
```
conda create -n ssdect python=3.7
conda activate ssdect
```

## Data Preparation
Use the NEU-DET as the dataset.
Split the dataset into training set and test set.
```
python neu_det_train/net_tool.py  
```

## Train
```
python tools/train.py  
```

## Test
```
python tools/test.py  
```