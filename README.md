# TAT-HHO: Tri-Attribute T-Net with Harris Hawks Optimization

**TAT-HHO** is a novel deep learning-based framework designed for enhanced **polyp segmentation** in gastrointestinal ultrasound endoscopy images. This repository implements a custom encoder-decoder architecture that integrates **Tri-Attribute Blocks (T-blocks)** and **Inception Multi-Scale Attention (IMA) modules**, trained using a carefully tuned SGD optimizer with momentum, early stopping, and learning rate scheduling. The architecture is further refined using **Harris Hawks Optimization** for hyperparameter tuning.

---

![Python](https://img.shields.io/badge/python-3.8+-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![Status](https://img.shields.io/badge/status-under%20review-yellow)


## 📁 Dataset

This implementation is tested using the **Kvasir-SEG** dataset:
- 1000 polyp images with corresponding segmentation masks
- Images are resized to **128×128 grayscale**
- Preprocessing includes mask binarization and image-mask mapping

### Dataset Structure (after preprocessing)
/Kvasir-SEG/
├── images/
└── masks/




---

## 🧠 Model Architecture

The **TAT-HHO** network has a dual-branch encoder with multi-scale convolutional blocks, followed by a high-capacity bottleneck and T-block based decoder. The key modules include:

- **T-Blocks**: Dual convolutional paths with residual concatenation for enriched spatial features
- **IMA Module**: Inception-style multi-scale attention fusion using 1×1, 3×3, and 5×5 convolutions
- **HHO Optimization**: Refines learning rate, kernel size, and number of filters
- **Skip Connections**: Preserve fine-grained boundaries essential for medical segmentation

> **Total Parameters**: 306,742,721  
> **Trainable Parameters**: 306,721,217  
> **Non-trainable Parameters**: 21,504

---

## 🔧 Training Details

- Optimizer: `SGD` with `momentum=0.9` and `nesterov=True`
- Loss: `Binary Cross Entropy + Dice Loss`
- EarlyStopping: patience = 15 epochs
- ReduceLROnPlateau: patience = 5 epochs, factor = 0.5
- Batch Size: 16  
- Epochs: 100  
- Learning Rate: Initially 0.01

---

## 📊 Performance

### 🔬 Statistical Summary from 15-Fold Cross-Validation

| Metric       | Mean      | Std Dev   | Skew      | Kurtosis  |
|--------------|-----------|-----------|-----------|-----------|
| Loss         | 0.1052    | 0.0021    | 0.0401    | -1.1518   |
| Accuracy     | 0.9557    | 0.0014    | -0.0388   | -0.8756   |
| mAP          | 0.9387    | 0.0011    | -0.1639   | -0.9464   |
| AUC          | 0.9664    | 0.0010    | 0.0373    | -1.0062   |
| Specificity  | 0.9937    | 0.0004    | -0.8630   | -0.0031   |
| Sensitivity  | 0.9766    | 0.0011    | -0.2115   | -0.8297   |

---

## 📈 Benchmarking with Existing Models

| Method (Backbone)                        | Jaccard ↑ | DSC ↑  | F2 ↑   | Recall ↑ | Accuracy ↑ |
|-----------------------------------------|-----------|--------|--------|----------|------------|
| FCN (Long et al., 2015)                 | 0.68±0.30 | 0.76   | 0.75   | 0.74     | 0.97       |
| U-Net (Ronneberger et al., 2015)        | 0.55±0.34 | 0.63   | 0.64   | 0.66     | 0.96       |
| PSPNet (Zhao et al., 2017)              | 0.72±0.27 | 0.80   | 0.79   | 0.79     | 0.98       |
| DeepLabV3+ (ResNet50)                   | 0.75±0.28 | 0.81   | 0.80   | 0.79     | 0.98       |
| ResNet-UNet (ResNet34)                  | 0.73±0.29 | 0.79   | 0.77   | 0.78     | 0.98       |
| DeepLabV3+ (ResNet101)                  | 0.75±0.28 | 0.82   | 0.80   | 0.81     | 0.98       |
| ResNet-UNet (ResNet101)                 | 0.74±0.29 | 0.80   | 0.80   | 0.80     | 0.98       |
| **TAT-HHO (Proposed)**                  | **0.7613** | **0.8247** | **0.967** | **0.976** | **0.99** |

---

## 🛠️ How to Run

### Setup
```bash
git clone https://github.com/yourusername/TAT-HHO.git
cd TAT-HHO
pip install -r requirements.txt
'''
## 📢 Note

📝 **This work is currently under peer review.**  
Once the paper is officially published, extended versions of the **TAT-HHO model**, including multiple architectural variants, training logs, and additional code utilities, will be made publicly available in future releases of this repository. Stay tuned!
