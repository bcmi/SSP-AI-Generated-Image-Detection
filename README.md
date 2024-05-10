# SSP-AI-Generated-Image-Detection

This is the official implementation for the following research paper:

> **A Single Simple Patch is All You Need for AI-generated Image Detection** [[arxiv]](https://arxiv.org/pdf/2402.01123.pdf)
>
> Jiaxuan Chen, Jieteng Yao, and Li Niu<br>

Note that in the paper, we proposed Enhanced SSP (ESSP) to improve its robustness against blur and compression. Currently, we only release the code of SSP. The code of ESSP will be released soon. 

<div align="center">
  <img src='./figures/pipeline.png' align="center" width=800>
</div>

## Environment setup
You can install the required packages by running the command:
```bash
pip install -r requirements.txt
```
## Dataset
The training set and testing set used in the paper can be downloaded in [GenImage](https://github.com/GenImage-Dataset/GenImage). This dataset contains data from eight generators. 
After downloading the dataset, you need to specify the root path in the options. The dataset can be organized as follows:
```bash
GenImage/
├── imagenet_ai_0419_biggan
    ├── train
        ├── ai
        ├── nature
    ├── val
        ├── ai
        ├── nature
└── imagenet_ai_0419_sdv4
    ├── train
        ├── ai
        ├── nature
    ├── val
        ├── ai
        ├── nature
└── imagenet_ai_0419_vqdm
    ...
└── imagenet_ai_0424_sdv5
    ...
└── imagenet_ai_0424_wukong
    ...
└── imagenet_ai_0508_adm
    ...
└── imagenet_glide
    ...
└── imagenet_midjourney
    ...
```
## Train and test
You can use this following command to train and test your model.
```bash
python train_val.py
``` 
Our pretrained models on sd4 can be downloaded in [Baidu Cloud](https://pan.baidu.com/s/1tmcfHeJfnlqcnqpZ_WtgBQ?pwd=7j1w)(code:7j1w)
