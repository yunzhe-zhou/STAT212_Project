## Description
STAT 212 project on data compression using neural networks. This project aims to make improvements based on [DeepZip] (https://arxiv.org/abs/1811.08162) by further compressing the prediction model.

Ref: [DeepZip: Lossless Data Compression using Recurrent Neural Networks](https://arxiv.org/abs/1811.08162)

## Requirements
0. GPU, nvidia-docker (or try alternative installation)
1. python 2/3
2. numpy
3. sklearn
4. keras 2.2.2
5. tensorflow (cpu/gpu) 1.8

(nvidia-docker is currently required to run the code)
A simple way to install and run is to use the docker files provided:

```bash
cd docker
make bash BACKEND=tensorflow GPU=0 DATA=/path/to/data/
```

## Alternative Installation
```bash
cd DeepZip
python3 -m venv tf
source tf/bin/activate
bash install.sh
```


## Code
To run a compression experiment: Directly run the file in src/main_code.ipynb
