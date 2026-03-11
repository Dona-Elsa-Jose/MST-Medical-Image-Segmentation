import numpy as np
import matplotlib.pyplot as plt
from skimage import io
import time
import os

# ==========================================
# 1. UNION-FIND DATA STRUCTURE
# ==========================================
class UnionFind:
    def __init__(self, size):
        self.parent = np.arange(size)
        self.size = np.ones(size)
        self.max_weight = np.zeros(size)

    def find(self, i):
        if self.parent[i] == i:
            return i
        self.parent[i] = self.find(self.parent[i])
        return self.parent[i]

    def union(self, i, j, weight):
        root_i = self.find(i)
        root_j = self.find(j)
        if root_i != root_j:
            if self.size[root_i] < self.size[root_j]:
                root_i, root_j = root_j, root_i
            self.parent[root_j] = root_i
            self.size[root_i] += self.size[root_j]
            self.max_weight[root_i] = max(self.max_weight[root_i], self.max_weight[root_j], weight)

# ==========================================
# 2. MST SEGMENTATION ALGORITHM
# ==========================================
def segment_image(img, k):
    R, C = img.shape
    edges = []
    # Build a graph where each pixel is a node and edges connect neighbors
    for r in range(R):
        for c in range(C):
            u = r * C + c
            if c + 1 < C:
                w = abs(float(img[r, c]) - float(img[r, c+1]))
                edges.append((w, u, u + 1))
            if r + 1 < R:
                w = abs(float(img[r, c]) - float(img[r+1, c]))
                edges.append((w, u, u + C))
    
    # Sort edges by weight (Kruskal's approach)
    edges.sort()
    uf = UnionFind(R * C)
    
    for w, u, v in edges:
        root_u, root_v = uf.find(u), uf.find(v)
        if root_u != root_v:
            # Felzenszwalb-Huttenlocher thresholding criterion
            threshold_u = uf.max_weight[root_u] + (k / uf.size[root_u])
            threshold_v = uf.max_weight[root_v] + (k / uf.size[root_v])
            if w <= min(threshold_u, threshold_v):
                uf.union(root_u, root_v, w)
    
    labels = np.array([uf.find(i) for i in range(R * C)]).reshape(R, C)
    return labels

# ==========================================
# 3. METRICS CALCULATION (MSE & PSNR)
# ==========================================
def get_metrics(original, denoised):
    mse = np.mean((original - denoised) ** 2)
    if mse == 0:
        return 0, 100.0
    psnr = 20 * np.log10(255.0 / np.sqrt(mse))
    return mse, psnr

# ==========================================
# 4. MAIN EXECUTION LOOP
# ==========================================
if __name__ == "__main__":
    # The 3 images from your dataset
    image_paths = [
        'axial-CT-image-of-lungs.png', 
        'image 2.png', 
        'image3.png'
    ]
    k_val = 500  # Matches the k-value from your friend's results

    print("... Initializing MST-based Image Processing ...")

    for path in image_paths:
        if not os.path.exists(path):
            print(f"\n❌ Skipping: {path} (File not found in folder)")
            continue

        print("\n" + "="*55)
        print(f"PROCESSING IMAGE: {path}")
        print("="*55)
        print(f"Segmenting with k={k_val} and applying denoising...")
        
        # Load image as grayscale
        img = io.imread(path, as_gray=True)
        if img.max() <= 1.0: 
            img = img * 255 # Normalize to 0-255 range
        
        start_time = time.time()
        
        # Perform segmentation
        labels = segment_image(img, k_val)
        
        # Create denoised version by averaging pixel intensities in each segment
        denoised = np.zeros_like(img)
        unique_labels, counts = np.unique(labels, return_counts=True)
        for label in unique_labels:
            mask = (labels == label)
            denoised[mask] = np.mean(img[mask])
            
        runtime = time.time() - start_time
        mse_val, psnr_val = get_metrics(img, denoised)

        # Console Output (Friend-Style)
        print("-" * 35)
        print("QUANTITATIVE METRICS")
        print("-" * 35)
        print(f"Computation Time:   {runtime:.2f} seconds")
        print(f"Number of Segments: {len(unique_labels)}")
        print(f"Mean Squared Error: {mse_val:.2f}")
        print(f"PSNR:               {psnr_val:.2f} dB")
        print("-" * 35)

        # Create 2x2 Visualization Grid
        plt.figure(figsize=(12, 8))
        plt.suptitle(f"MST Results: {path} (k={k_val})", fontsize=15)
        
        plt.subplot(2, 2, 1)
        plt.imshow(img, cmap='gray')
        plt.title('Original Medical Image')
        
        plt.subplot(2, 2, 2)
        plt.imshow(labels, cmap='nipy_spectral')
        plt.title('Segmentation Boundaries')
        
        plt.subplot(2, 2, 3)
        plt.imshow(denoised, cmap='gray')
        plt.title(f'Denoised (PSNR: {psnr_val:.2f})')
        
        plt.subplot(2, 2, 4)
        plt.hist(counts, bins=50, color='teal', edgecolor='black')
        plt.title('Segment Size Histogram')
        plt.xlabel('Pixels per Segment')
        plt.ylabel('Frequency')

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        
        # Save result to folder automatically
        out_name = f"result_{path.replace(' ', '_').split('.')[0]}.png"
        plt.savefig(out_name)
        print(f"💾 Visualization saved as: {out_name}")
        
        # IMPORTANT: Close this window to process the next image
        plt.show()