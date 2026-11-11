# Traffic Sign Recognition and Detection System

<p align="center">
  <strong>Deep Learning Capstone Project </strong><br>
  <strong>Department of Artificial Intelligence & Machine Learning (AIML)</strong><br>
  <strong>Faculty of Engineering and Technology (FET)</strong>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Guidance-IBM-052FAD?style=for-the-badge&logo=ibm&logoColor=white" alt="Guidance - IBM" />
  <img src="https://img.shields.io/badge/Deep%20Learning-PyTorch%20%7C%20TensorFlow-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white" alt="Framework" />
  <img src="https://img.shields.io/badge/Detector-YOLOv8-00FFFF?style=for-the-badge&logo=yolo&logoColor=black" alt="YOLOv8" />
  <img src="https://img.shields.io/badge/Accuracy-97.90%25-brightgreen?style=for-the-badge" alt="Accuracy" />
</p>

---

## 📌 Project Overview

This repository contains the official codebase, deep learning models, training scripts, graphical user interface, and technical capstone documentation for the **Traffic Sign Recognition and Detection System**, developed **under the guidance of IBM** for the course of **Deep Learning Project** in the **Department of FET AIML**.

Autonomous Vehicles (AVs) and Advanced Driver Assistance Systems (ADAS) depend critically on real-time visual perception to make life-critical navigational choices. Traffic signs communicate speed limits, directional instructions, right-of-way rules, and hazard alerts. This project delivers an end-to-end, two-stage perception stack designed to localize and categorize road signs under challenging real-world conditions (such as severe class imbalance, varying illumination, motion blur, and partial occlusions).

---

## 👥 Student Team & Authorship

This project was conceived, developed, and evaluated by the following team from the **Department of Artificial Intelligence & Machine Learning (AIML)**:

| S.No. | Candidate Name | University Roll Number | Department |
| :---: | :--- | :--- | :--- |
| **1** | **SHIVAM SHRESHTHA** | `FET-BAML-2024-28-084` | FET AIML |
| **2** | **K. DIGANT PATRO** | `FET-BAML-2024-28-082` | FET AIML |
| **3** | **AMRITANSH MANOJ SOHANI** | `FET-BAML-2024-28-083` | FET AIML |
| **4** | **ANSHUMAN NATH** | `FET-BAML-2024-28-085` | FET AIML |

---

## 🏛️ Industry Guidance & Mentorship

- **Industry Partner & Mentorship**: **IBM**
- **Course**: Deep Learning Project
- **Academic Unit**: Department of Artificial Intelligence & Machine Learning (AIML), Faculty of Engineering and Technology (FET)
- **Primary Domain**: Computer Vision, Autonomous Driving, Deep Convolutional Neural Networks, Object Detection

---

## 🏗️ System Architecture: The Two-Stage Perception Paradigm

Rather than relying on a monolithic detector that struggles with fine-grained multi-class classification on small bounding boxes, this system adopts a decoupled **Two-Stage Perception Paradigm**:

```
[ Input Video Feed / Camera Stream ]
                 │
                 ▼
     ┌───────────────────────┐
     │  Stage 1: YOLOv8      │  ──► Rapid Spatial Localization & Bounding Box Extraction
     │  (Trained on GTSDB)   │
     └───────────────────────┘
                 │
                 ▼
     ┌───────────────────────┐
     │  Region Preprocessing │  ──► Bounding Box Cropping, Resizing (32x32)
     │  & Normalization      │  ──► CLAHE Contrast Enhancement & Grayscale Mapping
     └───────────────────────┘
                 │
                 ▼
     ┌───────────────────────┐
     │  Stage 2: Custom CNN  │  ──► 6-Layer Deep CNN with Batch Normalization
     │  (Trained on GTSRB)   │  ──► Dropout Regularization & Softmax Classification
     └───────────────────────┘
                 │
                 ▼
     [ 43 Distinct Traffic Sign Classes + Confidence Score ]
                 │
                 ▼
     [ Multi-Threaded Real-Time GUI Display & Driver Warning ]
```

1. **Stage 1 (Localization & Detection)**: An optimized **YOLOv8** object detector trained on the **German Traffic Sign Detection Benchmark (GTSDB)** localizes traffic sign bounding boxes across arbitrary resolutions in real time at **28+ FPS**.
2. **Stage 2 (Fine-Grained Classification)**: Regions of interest (ROIs) are extracted, preprocessed using Contrast Limited Adaptive Histogram Equalization (CLAHE), and classified into **43 distinct categories** by a custom **6-Layer Deep Convolutional Neural Network (CNN)**.

---

## 📊 Datasets & Image Preprocessing

1. **German Traffic Sign Recognition Benchmark (GTSRB)**:
   - 43 distinct traffic sign classes (Speed limits, Prohibitions, Warnings, Mandatory directions).
   - 39,209 training instances and 12,630 test instances.
   - Significant natural class imbalance addressed through data augmentation.
2. **German Traffic Sign Detection Benchmark (GTSDB)**:
   - Full 1360&times;800 road scene photographs captured under varied natural lighting and weather conditions.
3. **Preprocessing Pipeline**:
   - **Grayscale Conversion**: Reduces dimensional complexity and mitigates color fading caused by weather.
   - **CLAHE (Contrast Limited Adaptive Histogram Equalization)**: Equalizes localized contrast, overcoming extreme backlighting, headlight glare, and shadowed forest roads.
   - **Data Augmentation**: Random geometric transformations (rotation &plusmn;10&deg;, random zoom 0.9–1.1&times;, translation &plusmn;10%) applied during training.

---

## 📈 Benchmark Performance & Experimental Results

The classification network was systematically refined across five architecture iterations:

| Model Iteration | Epochs | Batch Size | Learning Rate | Train Accuracy | Validation Accuracy | Validation Loss |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Model v1.0 | 5 | 64 | 0.001 | 91.20% | 90.85% | 0.3210 |
| Model v2.0 | 10 | 64 | 0.001 | 94.65% | 93.80% | 0.2150 |
| Model v3.0 | 10 | 32 | 0.0005 | 96.10% | 95.40% | 0.1540 |
| Model v4.0 | 10 | 32 | 0.0003 | 96.85% | 96.50% | 0.1210 |
| **Model v5.0 (Selected)** | **10** | **32** | **0.0002** | **97.73%** | **97.90%** | **0.0904** |

- **Final Classification Validation Accuracy**: **97.90%**
- **Final Classification Validation Loss**: **0.0904**
- **Inference Framerate**: **28–32 FPS** on standard hardware.

---

## 🖥️ Graphical User Interface (GUI)

The repository provides a multi-threaded desktop GUI (`gui.py`) implemented in Tkinter and OpenCV:
- **Image Mode**: Load arbitrary road images from disk to detect and identify signs with confidence percentages.
- **Video & Live Webcam Mode**: Processes live camera streams asynchronously on worker threads to eliminate UI freezing.
- **Dynamic Bounding Boxes**: Draws bounding overlays around identified road signs with class labels and confidence scores.

---

## 📄 Formal Academic Project Report

The complete 13-page academic capstone report prepared under the guidance of IBM is included directly in the root of this repository:
- **PDF Report**: [Traffic_Sign_Recognition_and_Detection_System_IBM_Report.pdf](Traffic_Sign_Recognition_and_Detection_System_IBM_Report.pdf)

---

## 🚀 Getting Started

### 1. Clone the Repository
```bash
git clone https://github.com/shivamshreshtha/Traffic-Sign-Recognition-and-Detection-System.git
cd Traffic-Sign-Recognition-and-Detection-System
```

### 2. Set Up Virtual Environment & Dependencies
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Launch the Application
- **Run the GUI Application**:
  ```bash
  python gui.py
  ```
- **Run Inference on Video / Image via CLI**:
  ```bash
  python main.py
  ```

---

## 🛠️ Technology Stack

- **Deep Learning Frameworks**: PyTorch, TensorFlow / Keras, Ultralytics YOLOv8
- **Computer Vision**: OpenCV (`cv2`), PIL (Pillow)
- **Data Engineering**: NumPy, Pandas, Scikit-learn, Matplotlib, Seaborn
- **User Interface**: Tkinter (Multi-threaded GUI)
- **Hardware Acceleration**: NVIDIA CUDA Toolkit & cuDNN

---

## 📜 Acknowledgements

We express our sincere appreciation to **IBM** for providing mentorship, real-world industry benchmarks, and strategic technical direction. We also extend our gratitude to the faculty and mentors of the **Department of Artificial Intelligence & Machine Learning (AIML), Faculty of Engineering and Technology (FET)** for their continued academic guidance and compute support.
