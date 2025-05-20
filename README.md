
# Nhóm 12: Truy xuất thông tin
## Thành viên:
```
B24CHKH004: Đặng Quang Dũng
B24CHHT013: Đỗ Hương Hà
B24CHHT014: Ngô Thị Mỹ Hà
B24CHHT040: Nguyễn Thu Thảo
B24CHHT031: Bùi Khắc Ngọc
```

# HRNet Image Classification with Imagenette2-160

This project uses [HRNet](https://github.com/HRNet/HRNet-Image-Classification) for image classification on the lightweight **Imagenette2-160** dataset.

# Experiments results:
## - **HRNet-W18**: 

Total Parameters: **21,299,004**
----------------------------------------------------------------------------------------------------------------------------------
Total Multiply Adds (For Convolution and Linear Layers only): **3.9893547743558884 GFLOPs**
----------------------------------------------------------------------------------------------------------------------------------
Number of Layers
* `Conv2d`: 325 layers
* `BatchNorm2d`: 325 layers
* `ReLU`: 284 layers
* `Bottleneck`: 8 layers
* `BasicBlock`: 104 layers
* `Upsample`: 31 layers
* `HighResolutionModule`: 8 layers
* `Linear`: 1 layers

=> loading model from `output/imagenette2-160/hrnet_w18_imagenette/model_best.pth.tar`

**Test Results:**

| Metric      | Value    |
| :---------- | :------- |
| Time        | 0.027    |
| Loss        | 0.3878   |
| Error@1     | 12.408   |
| Error@5     | 0.943    |
| Accuracy@1  | 87.592   |
| Accuracy@5  | 99.057   |

## - **HRNet-W32**:

## 📦 Installation

### ✅ 1. Install PyTorch Stable (2.7.0)

Follow the official [PyTorch Stable (2.7.0) installation guide](https://pytorch.org/get-started/previous-versions/) based on your system.

Example (with CUDA 12.8):

```bash
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```

### ✅ 2. Clone the HRNet repository

```bash
git clone https://github.com/HRNet/HRNet-Image-Classification
cd HRNet-Image-Classification
```

### ✅ 3. Install dependencies

```bash
pip install -r requirements.txt
```

Then edit the `requirements.txt` file:

```text
EasyDict==1.7
# opencv-python==3.4.1.15   ->  pip install opencv-python
# shapely==1.6.4            ->  pip install shapely>=1.8
Cython
scipy
pandas
pyyaml
json_tricks
scikit-image
yacs>=0.1.5
tensorboardX>=1.6
```

If not installed automatically:

```bash
pip install opencv-python shapely>=1.8
```

## 🖼️ Dataset: Imagenette2-160

Download and extract:

```bash
wget https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-160.tgz
tar -xzf imagenette2-160.tgz
```

### 🔍 What is Imagenette?

[Imagenette](https://github.com/fastai/imagenette) is a **lightweight subset of ImageNet**, consisting of 10 easily classified classes:

- Tench (a type of fish)
- English springer (a dog)
- Cassette player
- Chain saw
- Church
- French horn
- Garbage truck
- Gas pump
- Golf ball
- Parachute

This dataset is small (~13k images total), quick to train, and excellent for debugging and prototyping.

## ⚙️ Example Configuration

Path: `experiments/imagenette/hrnet_w18_imagenette.yaml`

```yaml
GPUS: (0,)
LOG_DIR: 'log/'
DATA_DIR: ''
OUTPUT_DIR: 'output/'
WORKERS: 4
PRINT_FREQ: 100

CUDNN:
  BENCHMARK: true
  DETERMINISTIC: false
  ENABLED: true

DATASET:
  DATASET: imagenette2-160
  ROOT: .
  DATA_FORMAT: jpg
  TRAIN_SET: train
  TEST_SET: val

MODEL:
  NAME: cls_hrnet
  NUM_CLASSES: 10
  IMAGE_SIZE:
    - 224
    - 224
  PRETRAINED: ''
  EXTRA:
    STAGE1:
      NUM_MODULES: 1
      NUM_BRANCHES: 1
      BLOCK: BOTTLENECK
      NUM_BLOCKS: [4]
      NUM_CHANNELS: [64]
      FUSE_METHOD: SUM
    STAGE2:
      NUM_MODULES: 1
      NUM_BRANCHES: 2
      BLOCK: BASIC
      NUM_BLOCKS: [4, 4]
      NUM_CHANNELS: [18, 36]
      FUSE_METHOD: SUM
    STAGE3:
      NUM_MODULES: 4
      NUM_BRANCHES: 3
      BLOCK: BASIC
      NUM_BLOCKS: [4, 4, 4]
      NUM_CHANNELS: [18, 36, 72]
      FUSE_METHOD: SUM
    STAGE4:
      NUM_MODULES: 3
      NUM_BRANCHES: 4
      BLOCK: BASIC
      NUM_BLOCKS: [4, 4, 4, 4]
      NUM_CHANNELS: [18, 36, 72, 144]
      FUSE_METHOD: SUM

TRAIN:
  BATCH_SIZE_PER_GPU: 32
  BEGIN_EPOCH: 0
  END_EPOCH: 100
  RESUME: false
  LR_FACTOR: 0.1
  LR_STEP: [30, 60, 90]
  OPTIMIZER: sgd
  LR: 0.01
  WD: 0.0001
  MOMENTUM: 0.9
  NESTEROV: true
  SHUFFLE: true

TEST:
  BATCH_SIZE_PER_GPU: 32
  MODEL_FILE: ''

DEBUG:
  DEBUG: false
```

Other configs:
- `experiments/imagenette/hrnet_w32_imagenette.yaml`
- `experiments/imagenette/hrnet_w48_imagenette.yaml`

```
# HRNet-W32:

NUM_CHANNELS: [32, 64]
...
NUM_CHANNELS: [32, 64, 128]
...
NUM_CHANNELS: [32, 64, 128, 256]

# HRNet-W48:
NUM_CHANNELS: [48, 96]
...
NUM_CHANNELS: [48, 96, 192]
...
NUM_CHANNELS: [48, 96, 192, 384]
```

## 🏋️‍♂️ Training

```bash
python tools/train.py --cfg experiments/imagenette/hrnet_w18_imagenette.yaml
```

## 🧪 Evaluation

```bash
python tools/valid.py --cfg experiments/imagenette/hrnet_w18_imagenette.yaml   TEST.MODEL_FILE output/imagenette/hrnet_w18/best.pth
```

## 📂 Folder Structure

```
HRNet-Image-Classification/
├── data/
├── lib/
├── log/
├── output/
├── experiments/
│   └── imagenette/
│       ├── hrnet_w18_imagenette.yaml
│       ├── hrnet_w32_imagenette.yaml
│       └── hrnet_w48_imagenette.yaml
├── tools/
│   ├── train.py
│   └── valid.py
└── imagenette2-160/
    ├── train/
    └── val/
```

## ✅ Notes

- ✅ PyTorch version Stable (2.7.0) is required — not compatible with newer versions.
- ✅ GPU training is highly recommended.
- ✅ Change `MODEL.PRETRAINED` in config if using pretrained weights.
