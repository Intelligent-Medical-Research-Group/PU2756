# PU2756: Lung Ultrasound Classification

This repository provides the **classification code** for the **PU2756** dataset—an open-access collection of lung ultrasound (LUS) images focused on automated benign–malignant analysis of lung tumors.

---

## 🧩 Overview

To the best of our knowledge, PU2756 is currently the **largest publicly available, expert-annotated lung-tumor ultrasound dataset**, and the **first** to jointly support **segmentation** and **classification** tasks. This resource enables deeper analysis of tumor characteristics and can help assess the necessity of fine-needle aspiration biopsy, offering practical value for ultrasound-based diagnostic decision-making.


---

## 🚀 Step 1: Image Segmentation

The **segmentation stage** is implemented in a separate module, which generates the region of interest (ROI) masks used for classification.  
You can find the segmentation code and detailed instructions in the following directory:

- 📁 **Folder:** [`segment/`](segment/)  
- 📘 **Documentation:** [Segmentation README](segment/README.md)




## 🧠 Step 2: Classification

After obtaining the segmented lung regions, the classification code in this repository can be used to:

- Load segmented or cropped ultrasound images  
- Extract features using deep neural networks  
- Train and evaluate classification models  
- Report accuracy and other performance metrics  

Detailed usage instructions and configuration files will be added soon.

---

## 📁 Repository Structure

