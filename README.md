# Military-Aircraft-Detection-EfficientNet-B3-Image-Classification
 【 EfficientNet-B3 Image Classification Framework with 5-Fold Cross Validation 】

This repository provides a **complete PyTorch-based image classification training and evaluation pipeline** using **EfficientNet-B3** with **5-Fold Cross Validation**.

The framework is designed for **research, academic projects, and reproducible experiments**, and is particularly suitable for **fine-grained visual classification tasks**, such as military aircraft recognition, remote sensing imagery, or other multi-class image classification problems.

This implementation is **primarily developed and validated using the Military Aircraft Detection Dataset from Kaggle**.

---

## 🚀 Features

* EfficientNet-B3 with ImageNet pretrained weights
* Class filtering via JSON configuration
* 5-Fold Cross Validation (K-Fold)
* Strong data augmentation strategy
* Automatic model checkpointing (per epoch & best model)
* Comprehensive evaluation metrics
* Rich visualization outputs for performance analysis

---

## 📁 Project Structure

```bash
.
├── Classifier_EfficientNet_b3_model.py   # Main training script
├── specified_classes.json               # Classes to include in training
├── crop/                                # Image dataset (ImageFolder format)
│   ├── class_A/
│   ├── class_B/
│   └── class_C/
└── /home/
    ├── FOLDA/
    ├── FOLDB/
    ├── FOLDC/
    ├── FOLDD/
    └── FOLDE/
```

---

## 🗂️ Primary Dataset

This project is **primarily designed and evaluated using the following Kaggle dataset**:

**Military Aircraft Detection Dataset (Kaggle)**
<img width="2246" height="299" alt="image" src="https://github.com/user-attachments/assets/b5c0ceaf-c5f1-4381-a7f5-4e9567cc583e" />

🔗 [https://www.kaggle.com/datasets/a2015003713/militaryaircraftdetectiondataset/data](https://www.kaggle.com/datasets/a2015003713/militaryaircraftdetectiondataset/data)

### Dataset Description

* Multi-class military aircraft image dataset
* Images collected under diverse backgrounds and viewpoints
* Suitable for fine-grained military aircraft classification

The dataset can be directly adapted to this framework by organizing images into **PyTorch ImageFolder format**:

```bash
crop/class_name/*.jpg
```

> ⚠️ **Important**: This repository focuses on **image classification**, not object detection.
> If the original dataset contains bounding boxes or annotations, images should be cropped or preprocessed beforehand.

---

## 🧠 Model Architecture

* **Backbone**: EfficientNet-B3 (ImageNet pretrained)
* **Classifier Head**: Fully Connected Layer (replaced)

```python
self.efficient_net = EfficientNet.from_pretrained('efficientnet-b3')
self.efficient_net._fc = nn.Linear(
    self.efficient_net._fc.in_features,
    num_classes
)
```

* **Loss Function**: CrossEntropyLoss
* **Optimizer**: Adam
* **Learning Rate**: `5e-5`

---

## 🖼️ Dataset Format

### 1. Directory Structure (PyTorch ImageFolder)

```bash
crop/
├── class_1/
│   ├── img001.jpg
│   ├── img002.jpg
├── class_2/
│   ├── img001.jpg
└── class_3/
```

### 2. Class Selection (`specified_classes.json`)

```json
[
  "class_1",
  "class_2",
  "class_3"
]
```

> ⚠️ Only classes listed in this JSON file will be included in training. All other classes are automatically excluded.

---

## 🔄 Data Preprocessing & Augmentation

```python
transforms.Resize((224, 224))
transforms.RandomHorizontalFlip()
transforms.RandomRotation(15)
transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1)
transforms.ToTensor()
transforms.Normalize(mean=[0.485, 0.456, 0.406],
                     std=[0.229, 0.224, 0.225])
```

✔ Compatible with ImageNet pretrained weights

---

## 🔁 Training Strategy

* **Cross Validation**: 5-Fold (KFold, shuffle=True, random_state=42)
* **Epochs**: 30
* **Batch Size**: 64
* **Device Selection**:

```python
device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
```

---

## 📊 Evaluation Metrics (per Epoch)

| Metric      | Description                     |
| ----------- | ------------------------------- |
| Accuracy    | Overall classification accuracy |
| Precision   | Weighted precision              |
| Recall      | Weighted recall                 |
| F1-score    | Weighted F1-score               |
| Sensitivity | TP / (TP + FN) per class        |
| Specificity | TN / (TN + FP) per class        |

---

## 📈 Visualization Outputs

For **each fold and each epoch**, the following outputs are generated automatically:

* Confusion Matrix
* ROC Curve (One-vs-Rest + Mean ROC with AUC)
* Training & Validation Loss Curve
* Training & Validation Accuracy Curve

Example filenames:

```bash
confusion_matrix_fold_1_epoch_3.png
roc_curve_fold_1_epoch_3.png
loss_curve_fold_1.png
accuracy_curve_fold_1.png
```

---

## 💾 Model Checkpoints & Logs

### Model Weights

```bash
model_epoch_1.pth
model_epoch_2.pth
...
best_model.pth
```

* `best_model.pth` is automatically selected based on the **highest validation accuracy**.

### Metrics (JSON)

```json
{
  "epoch": 3,
  "train_loss": 0.42,
  "train_accuracy": 85.3,
  "val_loss": 0.55,
  "val_accuracy": 82.1,
  "precision": 0.83,
  "recall": 0.82,
  "f1_score": 0.82,
  "sensitivity": [...],
  "specificity": [...]
}
```

---

## ▶️ How to Run

```bash
python Classifier_EfficientNet_b3_model.py
```

Before running, ensure that:

* The `crop/` dataset directory exists
* `specified_classes.json` is correctly configured
* All required Python packages are installed

---

## 📦 Requirements

```txt
torch
torchvision
efficientnet-pytorch
numpy
scikit-learn
matplotlib
seaborn
tqdm
```

Recommended environment:

* Python ≥ 3.8
* CUDA-enabled GPU for efficient training

---

## 🔧 Configurable Parameters

```python
EPOCHS = 30
batch_size = 64
learning_rate = 0.00005
n_splits = 5
```

---

## ⚠️ Notes

* ROC curves are computed using a **One-vs-Rest** strategy
* If a class is absent in a validation fold, ROC computation for that class may be undefined
* Weighted metrics are used to alleviate class imbalance

---

## 📜 License

This project is intended for **research and academic use only**.

Please ensure proper validation before any production or real-world deployment.

---

## ✉️ Experimental Results

<img width="1750" height="932" alt="image" src="https://github.com/user-attachments/assets/a5bfc257-e54e-4e11-8236-9568798a243e" />
<img width="1772" height="275" alt="image" src="https://github.com/user-attachments/assets/daf3da92-e1a8-4db8-a27a-6c5ab835d16d" />



⭐ If you find this repository useful, please consider starring it on GitHub!
