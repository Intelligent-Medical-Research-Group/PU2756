# PU2756: Lung Ultrasound Classification

This repository provides the **classification code** for the **PU2756** dataset—an open-access collection of lung ultrasound (LUS) images focused on automated benign–malignant analysis of lung tumors.

---

## 🧩 Overview

To the best of our knowledge, PU2756 is currently the **largest publicly available, expert-annotated lung-tumor ultrasound dataset**, and the **first** to jointly support **segmentation** and **classification** tasks. This resource enables deeper analysis of tumor characteristics and can help assess the necessity of fine-needle aspiration biopsy, offering practical value for ultrasound-based diagnostic decision-making.


---

## 🚀 Step 1: Segmentation

The **segmentation stage** is implemented in a separate module, which generates the region of interest (ROI) masks used for classification.  
You can find the segmentation code and detailed instructions in the following directory:

- 📁 **Folder:** [`segment/`](segment/)  
- 📘 **Documentation:** [Segmentation README](segment/README.md)




## 🧠 Step 2: Classification


This repository provides multiple **classification approaches** for the PU2756 lung ultrasound dataset, organized into separate folders for clarity and reproducibility.

The implemented methods include:

- **Radiomics + XGBoost**
- **Autoencoder + Radiomics + XGBoost**
- **Deep Learning–Based Models**

Each model type is implemented in an individual subfolder for modular management and easy benchmarking.


## 📁 Directory Structure


```text
PU2756/
├── RADIOMICS/      # Radiomics feature extraction + XGBoost classification
│
├── AE/      # Autoencoder feature extraction + XGBoost classification
│
├── MIX/            # Autoencoder feature learning + Radiomics + XGBoost classification
│
├── CNN/            # Deep learning model: for example ResNet

```


## ⚙️ How to Run

### 1️⃣ Prepare Environment and Dependencies

Make sure you have **Python ≥ 3.8** installed.

Install all necessary third-party libraries listed in **`requirements.txt`**:

```bash
pip install -r requirements.txt
```

---

### 2️⃣ Example: Train a Deep Learning Model


```bash
python main.py
```





✅ **Tip:**  
Make sure the segmentation results (ROI masks or cropped images) are ready before classification.  
You can find the segmentation module and instructions in:

- Folder: `segment/`
- Documentation: `segment/README.md`

