import os
import numpy as np
from PIL import Image

# =====================================================================
# 1. USER SETTINGS AND HYPERPARAMETERS
# =====================================================================

INPUT_HEIGHT = 64      # 64x64 as per dataset image for fast testing
INPUT_WIDTH = 64      
IN_CHANNELS = 3        # RGB channels (3)

# Conv2D (Convolution) Parameters
OUT_CHANNELS = 8       # Number of filters (feature maps)
KERNEL_SIZE = 3        # Filter size (e.g., 3x3)
CONV_STRIDE = 1        # Filter step size

# MaxPool2D (Pooling) Parameters
POOL_SIZE = 2          # Pooling window size (e.g., 2x2)
POOL_STRIDE = 2        # Pooling step size


# =====================================================================
# 2. DATASET LOADER (READS 4 FOLDERS FROM REPO)
# =====================================================================
data = []
labels = []

print("--- Loading Dataset from Folders ---")
for class_idx in range(4):
    folder_name = os.path.join("dataset", f"group_{class_idx}")
    
    if os.path.exists(folder_name):
        # Fetch up to 100 images per group
        image_files = [f for f in sorted(os.listdir(folder_name)) 
                       if f.lower().endswith(('png', 'jpg', 'jpeg'))][:100]
        
        for img_file in image_files:
            img_path = os.path.join(folder_name, img_file)
            
            # Load, resize to target dimensions, and ensure RGB format
            img = Image.open(img_path).convert('RGB').resize((INPUT_WIDTH, INPUT_HEIGHT))
            
            # Convert to numpy array and normalize pixel values to [0, 1]
            img_array = np.array(img, dtype=np.float32) / 255.0
            
            data.append(img_array)
            labels.append(class_idx)
            
        print(f"Loaded {len(image_files)} images from '{folder_name}/'")
    else:
        print(f"Warning: Folder '{folder_name}/' not found! Please ensure it exists.")

# Convert lists to NumPy arrays
X_train = np.array(data)
y_train = np.array(labels)


# =====================================================================
# 3. LAYER CLASSES (OPTIMIZED & VERIFIED)
# =====================================================================

class Conv2D:
    def __init__(self, in_channels, out_channels, kernel_size, stride=1):
        self.stride = stride
        # Filter shape: (out_channels, kernel_height, kernel_width, in_channels)
        self.filters = np.random.randn(out_channels, kernel_size, kernel_size, in_channels) * 0.1
        
        # FIXED: Initialized as a 1D array so self.biases[f] acts cleanly as a scalar
        self.biases = np.zeros(out_channels)

    def forward(self, X):
        self.X = X
        N, H, W, C = X.shape
        F, KK, _, _ = self.filters.shape
        
        # Validate dimensions
        if H < KK or W < KK:
            raise ValueError(f"Image dimensions ({H}x{W}) are smaller than kernel size ({KK}x{KK})!")

        out_h = (H - KK) // self.stride + 1
        out_w = (W - KK) // self.stride + 1
        
        out = np.zeros((N, out_h, out_w, F))
        for f in range(F):
            for i in range(out_h):
                for j in range(out_w):
                    h_start = i * self.stride
                    w_start = j * self.stride
                    # Extract patch -> Shape: (N, KK, KK, C)
                    patch = X[:, h_start:h_start+KK, w_start:w_start+KK, :]
                    
                    # Element-wise multiplication, sum over spatial/channel axes, add scalar bias
                    out[:, i, j, f] = np.sum(patch * self.filters[f], axis=(1, 2, 3)) + self.biases[f]
        return out

class ReLU:
    def forward(self, X):
        self.X = X
        return np.maximum(0, X)

class MaxPool2D:
    def __init__(self, pool_size=2, stride=2):
        self.pool_size = pool_size
        self.stride = stride

    def forward(self, X):
        self.X = X
        N, H, W, C = X.shape
        ps = self.pool_size
        stride = self.stride
        
        out_h = (H - ps) // stride + 1
        out_w = (W - ps) // stride + 1
        
        out = np.zeros((N, out_h, out_w, C))
        for i in range(out_h):
            for j in range(out_w):
                h_start = i * stride
                w_start = j * stride
                # Extract pooling window -> Shape: (N, ps, ps, C)
                patch = X[:, h_start:h_start+ps, w_start:w_start+ps, :]
                
                # Max value across spatial dimensions (height and width) -> Shape: (N, C)
                out[:, i, j, :] = np.max(patch, axis=(1, 2))
        return out


# =====================================================================
# 4. EXECUTION PIPELINE
# =====================================================================
if __name__ == "__main__":
    if X_train.size > 0:
        print(f"\n--- NumPy CNN Pipeline Started ---")
        print(f"Dataset Shape: {X_train.shape}")
        print(f"Labels Shape:  {y_train.shape}\n")

        # Define layers
        conv_layer = Conv2D(in_channels=IN_CHANNELS, out_channels=OUT_CHANNELS, 
                            kernel_size=KERNEL_SIZE, stride=CONV_STRIDE)
        relu_layer = ReLU()
        pool_layer = MaxPool2D(pool_size=POOL_SIZE, stride=POOL_STRIDE)

        # Forward pass verification
        out_conv = conv_layer.forward(X_train)
        print(f"Conv2D Output Shape:    {out_conv.shape}")

        out_relu = relu_layer.forward(out_conv)
        print(f"ReLU Output Shape:      {out_relu.shape}")

        out_pool = pool_layer.forward(out_relu)
        print(f"MaxPool2D Output Shape: {out_pool.shape}")
        
        print("\n--- Pipeline Completed Successfully ---")
    else:
        print("\nError: No images were loaded. Please verify that folders 'group_0' through 'group_3' exist.")
