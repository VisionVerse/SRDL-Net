# Structure-compensated Reliable Discriminability Learning for Cloud-Fog Degraded Remote Sensing Change Detection


<p align="center">
  <img src="https://img.shields.io/badge/Scene-Foggy%20Remote%20Sensing-green">
  <img src="https://img.shields.io/badge/Task-Remote%20Sensing%20Change%20Detection-purple">
  <img src="https://img.shields.io/badge/Python-3.10-blue">
  <img src="https://img.shields.io/badge/PyTorch-2.10-red">
</p>



## :pushpin: Introduction

This repository provides the implementation of a **Structure-compensated Reliable Discriminability Learning Network (SRDL-Net)** for **Cloud-Fog Degraded Remote Sensing Change Detection**.

Fog interference usually weakens structural cues of real changed regions and induces pseudo-change responses in unchanged backgrounds. To address this issue, SRDL-Net restores change discriminability by compensating fog-weakened structural cues and disentangling fog-induced pseudo-change responses during progressive change decoding.



<p align="center">
  <img src="figures/Framework.jpg" width="900">
</p>

SRDL-Net consists of the following components:

- **Structural Cue Compensation Module (SCCM)**  
  Compensates fog-weakened structural cues through spatial semantic preservation and wavelet-domain structural modeling.

- **Structural Edge Guidance (SEG)**  
  Aggregates multi-scale structure-enhanced features to generate edge guidance for boundary-aware decoding.

- **Degradation-induced Pseudo-change Suppression Module (DPSM)**  
  Suppresses fog-induced pseudo-change responses and progressively decodes reliable changed regions.

---

## :rocket: Installation

```text
SRDL-Net/
├── network/
│   ├── SRDL-Net.py
│   ├── SCC.py
│   ├── DPSM.py
│   ├── Edge.py
│   ├── cd_tools.py
│   └── backbones/
│       └── pvtv2.py
├── utils/
│   ├── dataloader.py
│   ├── metrics.py
│   └── tools.py
├── pretrained_model/
│   └── pvt_v2_b2.pth
├── figures/
│   └── Framework.jpg
├── train.py
├── requirements.txt
└── README.md
```


```bash
git https://github.com/VisionVerse/SRDL-Net.git
cd SRDL-Net

conda create -n ournet python=3.8
conda activate ournet

pip install -r requirements.txt
```

Please install a PyTorch version compatible with your CUDA version.
A typical environment includes:

```text
python >= 3.10
torch >= 2.10
torchvision
numpy
opencv-python
tqdm
scikit-learn
Pillow
```





SRDL-Net adopts **PVT-v2-B2** as the weight-sharing backbone. 
Please download the pretrained PVT-v2-B2 model and place it under:

```text
./pretrained_model/pvt_v2_b2.pth
```

The default path is:

```python
path = './pretrained_model/pvt_v2_b2.pth'
```


## :open_file_folder: Dataset Preparation

The new CFD RSCD datasets can be obtained from the [Cloud Drive](https://pan.jiangnan.edu.cn/link/AA0298DDB05F1E43F6A70CF8A4D664AAA3) [PW: RSCD].

Please organize the dataset as follows:

```text
CFD RSCD Dataset/
├── train/
│   ├── A/
│   ├── B/
│   └── label/
└── test/
    ├── A/
    ├── B/
    └── label/
```

Each sample contains:

- `A`: image at the first time point
- `B`: image at the second time point
- `label`: binary change mask

The ground-truth mask should follow:

```text
0: unchanged
1: changed
```

---

## :hourglass_flowing_sand: Training

Modify the dataset path and training configuration in `train_v2.py`, then run:

```bash
python train.py \
  --data_name CFD-LEVIR-CD \
  --epoch 200 \
  --batchsize 32 \
  --trainsize 256 \
  --lr 1e-4 \
  --edge_weight 0.1 \
  --reg_weight 1e-4
```

Main options:

```text
--data_name       dataset name
--epoch           number of training epochs
--batchsize       batch size
--trainsize       input image size
--lr              learning rate
--edge_weight     weight of edge supervision
--reg_weight      weight of regularization
```

The trained model will be saved to:

```text
./train_output/SRDL-Net/{data_name}/
```



## :bar_chart: Testing

After training, run:

```bash
python test.py \
  --data_name CFD-LEVIR-CD \
  --model_path ./train_output/SRDL-Net/CFD-LEVIR-CD/Seg_epoch_best.pth
```

The predicted change maps will be saved in the configured output directory.

---

## :bookmark_tabs: Citation

If you find this repository useful, please consider citing our paper:

```bibtex
@article{srdl2026zhou,
  title={Structure-compensated Reliable Discriminability Learning for Cloud-Fog Degraded Remote Sensing Change Detection},
  author={},
  journal={},
  year={2026}
}
```


## Acknowledgement
### :clap::clap::clap: Thanks to the authors of remote sensing change detection for their excellent works！



