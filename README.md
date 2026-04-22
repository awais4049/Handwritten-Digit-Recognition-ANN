# Handwritten Digit Recognition using ANN

A machine learning project that recognizes handwritten digits (0–9) using an Artificial Neural Network (ANN) built with TensorFlow/Keras and Scikit-learn.

> **Course:** Machine Learning — Mid-Term Project  
> **University:** COMSATS University Islamabad  
> **Accuracy Achieved:** 97.50%

---

## Overview

This project trains a fully connected ANN on the Scikit-learn Digits dataset to classify handwritten digits from 0 to 9. It covers the complete ML pipeline — data loading, preprocessing, visualization, model building, training, and evaluation.

---

## Dataset

- **Source:** `sklearn.datasets.load_digits`
- **Samples:** 1,797
- **Image Size:** 8×8 pixels (64 features)
- **Classes:** 10 (digits 0–9)
- **Pixel Range:** 0–16
- **Class Balance:** ~180 samples per class

---

## Project Structure

```
handwritten-digit-recognition-ann/
│
├── digit_recognition.ipynb   # Main Jupyter notebook (all steps)
├── digit_recognition.py      # Python script version
├── README.md                 # Project documentation
└── report/
    └── ML_Project_Report.docx
```

---

## ML Pipeline

| Step | Description |
|------|-------------|
| 1 | Load dataset using `load_digits()` |
| 2 | Visualize sample images, class distribution, pixel histogram |
| 3 | Normalize pixel values to [0, 1] |
| 4 | Split into 80% train / 20% test |
| 5 | Build ANN with 2 hidden layers |
| 6 | Train for 50 epochs with Adam optimizer |
| 7 | Evaluate with accuracy, confusion matrix, precision & recall |
| 8 | Interactive digit predictor |

---

## Model Architecture

```
Input (64)  →  Dense(128, ReLU)  →  Dense(64, ReLU)  →  Dense(10, Softmax)
```

| Layer | Neurons | Activation | Parameters |
|-------|---------|------------|------------|
| Hidden 1 | 128 | ReLU | 8,320 |
| Hidden 2 | 64 | ReLU | 8,256 |
| Output | 10 | Softmax | 650 |
| **Total** | | | **17,226** |

---

## Results

| Metric | Value |
|--------|-------|
| Test Accuracy | **97.50%** |
| Test Loss | 0.0988 |
| Macro Avg Precision | 0.98 |
| Macro Avg Recall | 0.98 |
| Macro Avg F1-Score | 0.98 |

### Per Class Performance

| Digit | Precision | Recall | F1-Score |
|-------|-----------|--------|----------|
| 0 | 0.97 | 0.97 | 0.97 |
| 1 | 1.00 | 1.00 | 1.00 |
| 2 | 0.97 | 1.00 | 0.99 |
| 3 | 0.97 | 0.97 | 0.97 |
| 4 | 0.98 | 1.00 | 0.99 |
| 5 | 0.96 | 0.96 | 0.96 |
| 6 | 0.97 | 0.97 | 0.97 |
| 7 | 1.00 | 0.97 | 0.99 |
| 8 | 0.97 | 0.97 | 0.97 |
| 9 | 0.97 | 0.95 | 0.96 |

---

## Installation

```bash
git clone https://github.com/your-username/handwritten-digit-recognition-ann.git
cd handwritten-digit-recognition-ann
pip install numpy matplotlib scikit-learn tensorflow
```

---

## Usage

```bash
# Run as Python script
python digit_recognition.py

# Or open the notebook
jupyter notebook digit_recognition.ipynb
```

### Predict a specific digit

```python
predict_digit(0)   # Shows image, predicted label, actual label, confidence %
predict_digit(42)
predict_digit(100)
```

---

## Technologies Used

![Python](https://img.shields.io/badge/Python-3.x-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.x-green)
![NumPy](https://img.shields.io/badge/NumPy-latest-lightblue)
![Matplotlib](https://img.shields.io/badge/Matplotlib-latest-red)

---

## License

This project is for academic purposes only.
