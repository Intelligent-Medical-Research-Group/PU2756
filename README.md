# PU2756: Lung Ultrasound Classification

This repository provides the **classification code** for the **PU2756** dataset—an open-access collection of lung ultrasound (LUS) images focused on automated benign–malignant analysis of lung tumors.

---

## 🧩 Overview

Accurate diagnosis of lung tumors from ultrasound is highly valuable yet challenging. Deep learning methods can assist clinicians, but their performance depends critically on the **quantity and quality of training data**. Due to the relatively limited use of LUS for benign–malignant diagnosis in routine practice, there has been **no public dataset** tailored to this task.

To bridge this gap, we release **PU2756**, an **open-access ultrasound dataset** for **both segmentation and classification** of lung tumors. PU2756 contains **3,860 B-mode images**, **expert annotations**, and **pathology-verified** labels. We also provide dataset statistics, baseline methods for segmentation and classification, and corresponding evaluation results to facilitate fair benchmarking.

To the best of our knowledge, PU2756 is currently the **largest publicly available, expert-annotated lung-tumor ultrasound dataset**, and the **first** to jointly support **segmentation** and **classification** tasks. This resource enables deeper analysis of tumor characteristics and can help assess the necessity of fine-needle aspiration biopsy, offering practical value for ultrasound-based diagnostic decision-making.


---

## 🚀 Step 1: Image Segmentation

The **segmentation stage** is implemented in a separate module, which generates the region of interest (ROI) masks used for classification.  
You can find the segmentation code and detailed instructions in the following directory:

👉 [Segmentation Code and README](/segment/)  




## 🧠 Step 2: Classification

After obtaining the segmented lung regions, the classification code in this repository can be used to:

- Load segmented or cropped ultrasound images  
- Extract features using deep neural networks  
- Train and evaluate classification models  
- Report accuracy and other performance metrics  

Detailed usage instructions and configuration files will be added soon.

---

## 📁 Repository Structure

