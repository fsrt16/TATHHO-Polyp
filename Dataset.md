# 📊 Dataset: Kvasir-SEG – Polyp Segmentation

## 📁 Overview

The **Kvasir-SEG** dataset is a publicly available, high-quality polyp segmentation dataset developed for machine learning research in gastrointestinal endoscopy. It contains 1000 images and their corresponding binary segmentation masks.

- **Total Samples**: 1000
- **Image Dimensions**: From 332×487 up to 1920×1072 pixels
- **Image Format**: JPEG (`.jpg`)
- **Mask Format**: Binary mask (`.jpg`, grayscale)
- **Classes**: Binary — Polyp (foreground), Background

## 🗂 Directory Structure

Kvasir-SEG/
├── images/
│ ├── image_001.jpg
│ ├── ...
├── masks/
│ ├── image_001.jpg
│ ├── ...
└── image_mask_mapping.csv
- Each image has a corresponding mask file with the **same filename**.
- The CSV file maps image-mask pairs used in preprocessing.

## 🧾 Annotation and Mask Generation

- **Annotation Tool**: [Labelbox](https://labelbox.com)
- **Process**:
  - Medical experts manually labeled all polyp regions.
  - Masks were generated using polygon coordinates exported from Labelbox.
  - White pixels (value = 255) represent the polyp; black (value = 0) represents background.
  - Superfluous elements (e.g., Olympus ScopeGuide markers) were removed and replaced with black patches.

## 🧪 Applications

Kvasir-SEG is suitable for:

- Semantic segmentation of polyps
- Computer-aided detection and diagnosis (CADx)
- Training deep learning models (e.g., U-Net, T-Net, FCNs, ViT)
- Evaluation of localization and detection performance
- Multi-dataset training for domain generalization

## ⚙️ Suggested Evaluation Metrics

The following metrics are commonly used to evaluate performance on this dataset:

| Metric           | Purpose                                     |
|------------------|---------------------------------------------|
| **Dice Score**   | Measures the overlap between prediction and ground truth |
| **IoU (Jaccard)**| Ratio of intersection over union            |
| **Accuracy**     | Overall pixel-wise correctness              |
| **Recall**       | True positive rate                          |
| **Specificity**  | True negative rate                          |
| **F2 Score**     | Harmonic mean, weighted towards recall      |

> It is encouraged to use multiple metrics for a holistic performance evaluation.

## 🧮 Dataset Statistics

- **Resolution**: Varies widely; no fixed resolution
- **Mask Coverage**: Varies per image
- **Foreground Classes**: Single class (polyp)

## 🧠 Use Cases

- Develop and benchmark segmentation models
- Train real-time polyp detection systems for colonoscopy
- Validate new architectures on standardized polyp data
- Perform ablation studies for clinical applications

## 📜 Citation

Please cite the original paper if you use this dataset:

```bibtex
@article{jha2020kvasir,
  title={Kvasir-SEG: A segmented polyp dataset},
  author={Jha, Debesh and Smedsrud, Pia H and Riegler, Michael A and Halvorsen, P{\aa}l and de Lange, Thomas and Johansen, Dag and Pettersen, Svein A},
  journal={Proceedings of International Conference on Multimedia Modeling},
  year={2020},
  publisher={Springer}
}


🔗 License
The dataset is released under an open-access license and can be freely used for academic and research purposes.

✅ Quick Facts
Key	Value
Number of Samples	1000
Classes	Polyp, Background
Annotations	Manual
Tool Used	Labelbox
Source	Kvasir v2
