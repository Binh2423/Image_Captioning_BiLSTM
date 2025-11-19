# Image Captioning with Transformer-Based Architecture

## Overview

This project is part of a university course on Natural Language Processing (NLP). The objective is to develop a model that can generate captions for images using a transformer-based architecture. We have utilized a Data-efficient Image Transformer (DeiT) for the encoder and a standard transformer decoder. The Flickr30k dataset was used for training, with additional text preprocessing and image resizing and augmentation techniques applied,for more details see : [Report](https://github.com/Devnetly/image-captioning/blob/main/docs/report.pdf).


<img src="https://github.com/Devnetly/image-captioning/blob/main/docs/figures/architecture-with-bg.png?raw=true" alt="drawing" width="100%"/>

## Members

- Abdelnour Fellah: [ab.fellah@esi-sba.dz](mailto:ab.fellah@esi-sba.dz)
- Abderrahmane Benounene: [a.benounene@esi-sba.dz](mailto:a.benounene@esi-sba.dz)
- Adel Abdelkader Mokadem: [aa.mokadem@esi-sba.dz](mailto:aa.mokadem@esi-sba.dz)
- Meriem Mekki: [me.mekki@esi-sba.dz](mailto:me.mekki@esi-sba.dz)
- Yacine Lazreg Benyamina: [yl.benyamina@esi-sba.dz](mailto:yl.benyamina@esi-sba.dz)

## Setup

To set up the project, please follow these steps:

```sh
git clone git@github.com:Devnetly/image-captioning.git
cd image-captioning
conda create -n automatic-image-captioning
conda activate automatic-image-captioning
pip install -r requirements.txt
```

Once the envirement is ready, run the `initialize.py` to split the dataset and create the vocabulary,initialy the /data folder structure should be like this : 

```
data
└── flickr30k
    ├── captions.csv
    ├── images
    │   └── 0.jpg
    │   └── 1.jpg
    │   ⋮
    │   └── n.jpg
```

Then run the command : 

```sh
python initialize.py --dataset {flickr30k} [--min-freq MIN_FREQ]
```

And the folder structure should then become something similar to the one below:

```
data
└── flickr30k
    ├── captions.csv
    ├── images
    │   └── 0.jpg
    │   └── 1.jpg
    │   ⋮
    │   └── n.jpg
    ├── test_captions.csv
    ├── train_captions.csv
    └── vocab.pkl
```

## Training

To train the model,follow the steps below : 

```sh
cd src/training
python train.py [-h] --dataset {flickr30k} [--batch-size BATCH_SIZE] [--learning-rate LEARNING_RATE] [--weight-decay WEIGHT_DECAY] [--epochs EPOCHS] [--num-workers NUM_WORKERS] [--prefetch-factor PREFETCH_FACTOR] --weights-folder WEIGHTS_FOLDER --histories-folder HISTORIES_FOLDER
```

## Inference

To generate caption for a set of images in folder,follow these steps : 

```sh
cd src/inference
python inference.py [-h] [--dataset {flickr30k}] [--model {transformer}] --checkpoint CHECKPOINT [--source SOURCE] --destination DESTINATION
```

## Autoregressive Image Captioning (New)

A new standalone script `scripts/caption_ar.py` provides an autoregressive decoder with attention and advanced features:

### Features

- **DeiT Feature Extractor**: Uses Data-efficient Image Transformer (DeiT) via timm with PyTorch weights
- **Linear Projection**: Projects DeiT features before the decoder to reduce parameters
- **Autoregressive Decoder**: Uni-directional decoder with attention mechanism for proper sequential generation
- **Weight Tying**: Shares weights between decoder embedding and output projection
- **Feature Pre-extraction**: Save DeiT features to disk for faster language model training
- **Configurable Architecture**: Adjustable dropout, number of layers, attention heads
- **Freezeable Backbone**: Option to freeze DeiT parameters to focus on caption generation

### Usage

#### Pre-extract Features (Optional, for faster training)

```sh
python scripts/caption_ar.py --mode preextract \
  --data-dir data/flickr30k \
  --train-captions data/flickr30k/train_captions.csv \
  --features-dir data/features \
  --pretrained
```

#### Training

With on-the-fly feature extraction:
```sh
python scripts/caption_ar.py --mode train \
  --data-dir data/flickr30k \
  --train-captions data/flickr30k/train_captions.csv \
  --output-dir output \
  --pretrained \
  --freeze-deit \
  --projection-dim 512 \
  --num-layers 6 \
  --num-heads 8 \
  --dropout 0.1 \
  --batch-size 32 \
  --epochs 10
```

With pre-extracted features (faster):
```sh
python scripts/caption_ar.py --mode train \
  --use-preextracted \
  --features-dir data/features \
  --train-captions data/flickr30k/train_captions.csv \
  --output-dir output \
  --projection-dim 512 \
  --num-layers 6 \
  --num-heads 8 \
  --dropout 0.1 \
  --batch-size 32 \
  --epochs 10
```

#### Inference

Single image:
```sh
python scripts/caption_ar.py --mode inference \
  --checkpoint output/final_model.pt \
  --image-path test_images/sample.jpg \
  --output-dir output \
  --projection-dim 512 \
  --num-layers 6 \
  --num-heads 8
```

Batch inference:
```sh
python scripts/caption_ar.py --mode inference \
  --checkpoint output/final_model.pt \
  --images-dir test_images/ \
  --output-dir output \
  --projection-dim 512 \
  --num-layers 6 \
  --num-heads 8
```

### Key Arguments

- `--mode {train,inference,preextract}`: Operating mode
- `--pretrained`: Use pretrained DeiT weights (PyTorch only)
- `--freeze-deit`: Freeze DeiT backbone parameters
- `--projection-dim`: Dimension of linear projection layer (default: 512)
- `--num-layers`: Number of decoder layers (default: 6)
- `--num-heads`: Number of attention heads (default: 8)
- `--dropout`: Dropout rate (default: 0.1)
- `--use-preextracted`: Use pre-extracted features for training
- `--features-dir`: Directory for pre-extracted features

## Run the associated app

To run the app associated with the project : 

```sh
cd app
streamlit run main.py
```