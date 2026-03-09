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
```

##⚙️ Installation & Setup
1. Clone the repository

Bash
git clone [https://github.com/YourUsername/Your-Repo-Name.git](https://github.com/YourUsername/Your-Repo-Name.git)
cd Your-Repo-Name
2. Environment Requirements
It is highly recommended to use Python 3.8 for native dlib pre-compiled wheel support on Windows.

Bash
pip install opencv-python numpy
Note for Windows users: To install dlib without C++ compilation errors (CMake), please download the dlib-19.19.0-cp38-cp38-win_amd64.whl file and install it directly via pip.

3. Download the Dlib Landmark Model
Download shape_predictor_68_face_landmarks.dat from the official Dlib repository or search online, and place it in the root directory of this project.

##🚀 Usage
Run the Evaluation Pipeline:
Modify the happy_dir and sad_dir paths in the script to match your local dataset location, then run:

Bash
python facial_detection.py
Generate Visual Demonstrations:
To see the algorithm in action with landmarks and metric texts drawn on the image:

Bash
python generate_demo.py

##📊 Experimental Results
Evaluated on a balanced subset of the Facial Expressions (Happiness, Sadness, Surprise) dataset from Kaggle, presenting extreme challenges like tight cropping, high variance in poses, and occlusions:

Overall Accuracy: 78.4%

Happy Class: Recall 85.9%, Precision 73.8%

Sad Class: Recall 67.7%, Precision 85.1%

Analysis shows that while explicit geometric heuristics offer a strong baseline, they are vulnerable to extreme feature overlap (e.g., crying vs. laughing) and lack the global texture receptive fields necessary to capture subtle micro-expressions.

##📜 Acknowledgments
Dataset sourced from Kaggle: Facial Expressions (Happiness, Sadness, Surprise)

Dlib implementation based on V. Kazemi and J. Sullivan's paper: "One millisecond face alignment with an ensemble of regression trees" (CVPR 2014).
