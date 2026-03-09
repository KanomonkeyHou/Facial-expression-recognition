# Facial Expression Recognition (Happy vs. Sad) via Geometric Heuristics

![Python](https://img.shields.io/badge/Python-3.8-blue.svg)
![OpenCV](https://img.shields.io/badge/OpenCV-4.x-green.svg)
![Dlib](https://img.shields.io/badge/Dlib-19.19-orange.svg)

## 📌 Project Overview
This repository contains the source code for an experimental Computer Vision project focused on **Facial Expression Recognition (FER)**. Unlike modern Deep Learning approaches that function as "black boxes" and require massive GPU computing power, this project implements a highly interpretable, **zero-training rule-based classification pipeline**.

By extracting 68 facial landmarks using Dlib and calculating specific explicit geometric features—such as Eye Aspect Ratio (EAR), Mouth Aspect Ratio (MAR), and Nose-Mouth Corner Ratio (NMR)—the system effectively classifies images into `Happy` or `Sad` categories.

## ✨ Key Features
* **Zero-Shot & CPU-Friendly**: No model training required. Real-time inference runs perfectly on a standard CPU.
* **High Interpretability**: Every classification decision can be traced back to explicit physical mathematical formulas (Euclidean distances).
* **Robust Fallback Mechanism**: Includes a custom bounding box injection strategy to handle tightly cropped, "in-the-wild" images where standard HOG detectors fail.
* **Visual Demonstration**: Built-in script to map the 68 landmarks onto the face and display real-time metric calculations on the image.

## 📂 Project Structure
```text
FER-Project/
│
├── facial_detection.py       # Main evaluation script for the dataset
├── generate_demo.py          # Script to generate visual demonstration images
├── README.md                 # Project documentation
│
├── shape_predictor_68_face_landmarks.dat  # Dlib pre-trained model (Must be downloaded manually)
│
└── archive/                  # Kaggle Dataset directory
    └── test/
        ├── happy/            # 92 images
        └── sad/              # 93 images
