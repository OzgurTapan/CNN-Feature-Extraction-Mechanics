# Pure NumPy Convolutional Neural Network (CNN) from Scratch

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![NumPy](https://img.shields.io/badge/Library-NumPy%20Only-orange)
![License](https://img.shields.io/badge/License-MIT-green)

A modular, highly customizable implementation of a Convolutional Neural Network (CNN) built **strictly using NumPy and standard Python libraries**. No heavy deep learning frameworks (like PyTorch or TensorFlow) are used under the hood. 

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
| **Input Data** | Loaded & resized RGB images | `(400, 300, 300, 3)` |
| **Conv2D** | $3\times3$ kernel, 8 filters, stride=1 | `(400, 298, 298, 8)` |
| **ReLU** | Element-wise non-linearity | `(400, 298, 298, 8)` |
| **MaxPool2D** | $2\times2$ pooling window, stride=2 | `(400, 149, 149, 8)` |

---

## 🛠️ Installation & Prerequisites

Make sure you have Python installed along with the required standard libraries:
```
pip install numpy pillow
```

---

## ⚙️ How to Run
1.  Clone the repository:

git clone [https://github.com/YOUR_USERNAME/numpy-cnn-from-scratch.git](https://github.com/YOUR_USERNAME/numpy-cnn-from-scratch.git)
cd numpy-cnn-from-scratch

   ```
   git clone [https://github.com/OzgurTapan/Modeling-of-Basic-Convolutional-Neural-Network.git](https://github.com/OzgurTapan/Modeling-of-Basic-Convolutional-Neural-Network.git) cd Modeling-of-Basic-Convolutional-Neural-Network
   ```
2.  Ensure your dataset folders (group_0, group_1, group_2, group_3) are placed in the root directory, each containing up to 100 images.
   
3.  Run the script:
    ```
    python cnn.py
    ```

---

## 🎛️ Hyperparameter Customization
You can tweak the network parameters directly at the top of cnn.py:

INPUT_HEIGHT = 300     # Image height (can be reduced to 64 for faster testing)  
INPUT_WIDTH = 300      
OUT_CHANNELS = 8       # Number of convolutional filters  
KERNEL_SIZE = 3        # Filter matrix size (e.g., 3x3)  
CONV_STRIDE = 1        # Filter step size  
POOL_SIZE = 2          # Max pooling window size  

---

## 📈 Future Improvements
[ ] Implement Backpropagation and Gradient Descent.  
[ ] Add Fully Connected (Dense) output layers with Softmax cross-entropy loss.  
[ ] Optimize spatial loops using im2col vectorization for faster execution.  
