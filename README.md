# Wind Turbine Blade Fault Classification Using a Simple CNN

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?style=flat-square&logo=python)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange?style=flat-square&logo=tensorflow)
![Keras](https://img.shields.io/badge/Keras-Deep%20Learning-red?style=flat-square&logo=keras)
![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)
![Status](https://img.shields.io/badge/Status-Complete-brightgreen?style=flat-square)

> Automatically classify wind turbine blade images as **Healthy** or **Faulty** using a Convolutional Neural Network (CNN) built with TensorFlow/Keras.

---

## Table of Contents

- [Project Overview](#-project-overview)
- [Dataset](#-dataset)
- [Project Structure](#-project-structure)
- [Model Architecture](#-model-architecture)
- [Results](#-results)
- [Requirements](#-requirements)
- [How to Run](#-how-to-run)
- [Step-by-Step LaTeX Files](#-step-by-step-latex-files)
- [Figures](#-figures)
- [Conclusion](#-conclusion)
- [Author](#-author)

---

## Project Overview

Wind turbine blades can develop defects such as **cracks**, **erosion**, and **surface damage**. Traditional manual inspection is time-consuming and requires expert knowledge.

This project solves the problem automatically using **image classification**:

| Input | Output |
|-------|--------|
| Image of a wind turbine blade | `Healthy` or `Faulty` |

**Key techniques used:**
- Image resizing and normalisation
- Data augmentation (rotation, flip, zoom, shift)
- CNN with 3 convolutional blocks
- Dropout regularisation
- Early stopping to prevent overfitting

---

## Dataset

| Property | Details |
|----------|---------|
| **Name** | CAI-SWTB Dataset |
| **Source** | Kaggle |
| **Link** | [CAI-SWTB on Kaggle](https://www.kaggle.com/datasets/mohammadshekaramiz/small-wind-turbine-blade-dataset-cai-swtb) |
| **Classes** | `Faulty`, `Healthy` |
| **Training images** | 4,200 (2,100 per class) |
| **Validation images** | 600 (300 per class) |
| **Test images** | 1,200 (600 per class) |

Images were collected in both **indoor** and **outdoor** environments.

---

## 📁 Project Structure

```
wind-turbine-blade-cnn/
│
├── tex_steps/                        # LaTeX source files (one per step)
│   ├── step01_import_libraries.tex
│   ├── step02_dataset_paths.tex
│   ├── step03_dataset_info.tex
│   ├── step04_sample_images.tex
│   ├── step05_preprocess_augment.tex
│   ├── step06_load_data.tex
│   ├── step07_build_model.tex
│   ├── step08_compile_model.tex
│   ├── step09_train_model.tex
│   ├── step10_evaluate_model.tex
│   ├── step11_plot_graphs.tex
│   ├── step12_confusion_matrix.tex
│   ├── step13_example_predictions.tex
│   └── step14_save_model.tex
│
├── simple_wind_turbine_cnn.h5        # Saved trained model (after running)
└── README.md                         # This file
```

---

## Model Architecture

```
Input (128 × 128 × 3)
        │
   Conv2D(32)  + ReLU
   MaxPooling2D
        │
   Conv2D(64)  + ReLU
   MaxPooling2D
        │
   Conv2D(128) + ReLU
   MaxPooling2D
        │
     Flatten
        │
   Dense(64)  + ReLU
   Dropout(0.5)
        │
   Dense(1)   + Sigmoid
        │
   Output: Healthy / Faulty
```

| Layer | Output Shape | Parameters |
|-------|-------------|------------|
| Conv2D (32 filters) | 126 × 126 × 32 | 896 |
| MaxPooling2D | 63 × 63 × 32 | 0 |
| Conv2D (64 filters) | 61 × 61 × 64 | 18,496 |
| MaxPooling2D | 30 × 30 × 64 | 0 |
| Conv2D (128 filters) | 28 × 28 × 128 | 73,856 |
| MaxPooling2D | 14 × 14 × 128 | 0 |
| Flatten | 25,088 | 0 |
| Dense (64) | 64 | 1,605,696 |
| Dropout (0.5) | 64 | 0 |
| Dense (1) | 1 | 65 |
| **Total** | | **1,699,009** |

---

## Results

| Metric | Faulty | Healthy | Overall |
|--------|--------|---------|---------|
| Precision | 0.72 | 0.74 | 0.73 |
| Recall | 0.76 | 0.71 | 0.73 |
| F1-Score | 0.74 | 0.73 | 0.73 |
| **Accuracy** | — | — | **73%** |

**Confusion Matrix (Test Set — 1,200 images):**

```
                Predicted Faulty    Predicted Healthy
Actual Faulty        454                 146
Actual Healthy       174                 426
```

---

## Requirements

Install all dependencies with:

```bash
pip install tensorflow scikit-learn matplotlib seaborn numpy
```

| Package | Version |
|---------|---------|
| Python | 3.8+ |
| TensorFlow | 2.x |
| scikit-learn | latest |
| matplotlib | latest |
| seaborn | latest |
| numpy | latest |

---

## How to Run

### 1. Clone the repository

```bash
git clone https://github.com/your-username/wind-turbine-blade-cnn.git
cd wind-turbine-blade-cnn
```

### 2. Download the dataset

Download the CAI-SWTB dataset from [Kaggle](https://www.kaggle.com/datasets/mohammadshekaramiz/small-wind-turbine-blade-dataset-cai-swtb) and place it in the following structure:

```
CAI-SWTB-Dataset/
├── Train/
│   ├── Faulty/
│   └── Healthy/
├── Validation/
│   ├── Faulty/
│   └── Healthy/
└── Test/
    ├── Faulty/
    └── Healthy/
```

### 3. Update the dataset paths

In **Step 2**, update the directory paths to match your local setup:

```python
train_dir = "/your/path/CAI-SWTB-Dataset/Train"
val_dir   = "/your/path/CAI-SWTB-Dataset/Validation"
test_dir  = "/your/path/CAI-SWTB-Dataset/Test"
```

### 4. Run the steps in order

Run each step file in sequence (Step 1 → Step 14). If using a Jupyter Notebook or Kaggle, paste the code from each `.tex` file's listing block into consecutive cells.

### 5. Load the saved model (optional)

```python
from tensorflow.keras.models import load_model
model = load_model("simple_wind_turbine_cnn.h5")
```

---

## Step-by-Step LaTeX Files

Each step of the project is documented in a separate `.tex` file inside the `tex_steps/` folder. These can be compiled individually using **Overleaf** or any local LaTeX distribution (`pdflatex`).

| File | Description |
|------|-------------|
| `step01_import_libraries.tex` | Import all required Python libraries |
| `step02_dataset_paths.tex` | Set paths to Train / Validation / Test folders |
| `step03_dataset_info.tex` | Count and print image counts per class |
| `step04_sample_images.tex` | Display a 3×3 grid of sample images |
| `step05_preprocess_augment.tex` | Define image generators with augmentation |
| `step06_load_data.tex` | Load data using `flow_from_directory` |
| `step07_build_model.tex` | Define the CNN architecture |
| `step08_compile_model.tex` | Compile with Adam + binary crossentropy |
| `step09_train_model.tex` | Train with early stopping |
| `step10_evaluate_model.tex` | Evaluate on test set |
| `step11_plot_graphs.tex` | Plot training/validation accuracy & loss |
| `step12_confusion_matrix.tex` | Generate confusion matrix and classification report |
| `step13_example_predictions.tex` | Visualise predictions on 9 test images |
| `step14_save_model.tex` | Save the trained model to `.h5` file |

---

## Figures

| Figure | Description |
|--------|-------------|
| Figure 1 | Sample images from the dataset (healthy & faulty) |
| Figure 2 | Training and Validation Accuracy over epochs |
| Figure 3 | Training and Validation Loss over epochs |
| Figure 4 | Confusion Matrix on the test set |
| Figure 5 | Example predictions on 9 test images |

---

## Conclusion

This project demonstrated that a simple CNN can effectively detect wind turbine blade faults from images, achieving **73% test accuracy** on a balanced dataset of 1,200 test images. The approach is:

- Simple and practical
- Suitable for undergraduate-level study
- Applicable to real-world renewable energy inspection problems

**Possible improvements:**
- Use transfer learning (e.g., ResNet50, MobileNetV2) for higher accuracy
- Increase training epochs with a lower learning rate
- Try larger image sizes (e.g., 224 × 224)
- Apply class activation mapping (CAM) for visual explainability

---

## Author

**Amreen Batool**


