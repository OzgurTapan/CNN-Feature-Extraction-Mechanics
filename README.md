# Pure NumPy Convolutional Neural Network (CNN) Feature Extraction Mechanics (Forward Pass Demo)

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![NumPy](https://img.shields.io/badge/Library-NumPy%20Only-orange)
![License](https://img.shields.io/badge/License-MIT-green)

A modular, highly customizable implementation of a Convolutional Neural Network (CNN) Feature Extractor built **strictly using NumPy and standard Python libraries**. No heavy deep learning frameworks (like PyTorch or TensorFlow) are used under the hood. 

The goal of this project is to demystify the mathematical operations, tensor transformations, and forward-pass mechanics that happen inside modern deep learning layers.

---

## 🚀 Key Features

* **Zero Framework Dependencies:** Uses only `numpy`, `os`, and `PIL` (for image loading and resizing).
* **Modular Layer Design:** Object-oriented implementation featuring clean, separate classes for `Conv2D`, `ReLU`, and `MaxPool2D`.
* **Fully Parametric Architecture:** Easily modify input image dimensions, filter sizes, stride lengths, and pooling windows via simple global configuration variables.
* **Automated Dataset Ingestion:** Automatically reads, resizes, and normalizes image data from 4 distinct class folders (`group_0` to `group_3`).

---

## 📐 Architecture Flow & Shape Transformation

When processing the dataset through the pipeline, the tensor shapes change dynamically:

| Layer | Operation / Formula | Output Shape |
| :--- | :--- | :--- |
| **Input Data** | Loaded & resized RGB images | `(400, 64, 64, 3)` |
| **Conv2D** | $3\times3$ kernel, 8 filters, stride=1 | `(400, 62, 62, 8)` |
| **ReLU** | Element-wise non-linearity | `(400, 62, 62, 8)` |
| **MaxPool2D** | $2\times2$ pooling window, stride=2 | `(400, 31, 31, 8)` |

---

## 🛠️ Installation & Prerequisites

Make sure you have Python installed along with the required standard libraries:
```
pip install numpy pillow
```

---

## ⚙️ How to Run

1.  Clone the repository:
```
git clone [https://github.com/OzgurTapan/CNN-Feature-Extractor-From-Scratch.git](https://github.com/OzgurTapan/CNN-Feature-Extractor-From-Scratch.git) cd CNN-Feature-Extractor-From-Scratch
```
2.  Ensure your dataset folder is placed in the root directory, containing 4 groups with a total of 400 images.
   
3.  Run the script:
```
python Basic_CNN.py
```

---

## 🎛️ Hyperparameter Customization
You can tweak the network parameters directly at the top of cnn.py:

INPUT_HEIGHT = 64      # Image height (can be reduced for faster testing)  
INPUT_WIDTH = 64      
OUT_CHANNELS = 8       # Number of convolutional filters  
KERNEL_SIZE = 3        # Filter matrix size (e.g., 3x3)  
CONV_STRIDE = 1        # Filter step size  
POOL_SIZE = 2          # Max pooling window size  

---

## 📈 Final Notes
Interestingly, in machine learning, even untrained random convolutional layers can be  used sometimes as a baseline feature extractor (or in techniques like extreme learning machines) because the local sliding window structure inherently detects edges and textures regardless of whether the weights are learned or random.
