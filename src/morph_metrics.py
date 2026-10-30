import pandas as pd
import numpy as np
import skimage as ski
from skimage import measure
import torch
import torch.nn.functional as F
from transformers import ViTImageProcessor, ViTModel
from PIL import Image
from scipy.fft import fft, ifft
from skimage.draw import ellipse
from skimage.measure import find_contours
import sklearn
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler




#Scalar metrics scripts
def scalar_morph(mask):
    scalar_metrics_exp = measure.regionprops_table(mask, properties=['label', 'area', 'perimeter', 'eccentricity', 'solidity', 'extent', 'orientation','major_axis_length', 'minor_axis_length'])
    scalar_metrics_exp = pd.DataFrame(scalar_metrics_exp).set_index('label')

    scalar_metrics_exp['circularity'] = (4 * np.pi * scalar_metrics_exp['area']) / (scalar_metrics_exp['perimeter'] ** 2)
    scalar_metrics_exp['aspect_ratio'] = scalar_metrics_exp['major_axis_length'] / (scalar_metrics_exp['minor_axis_length'] + 1e-6)  # Avoid division by zero

    return scalar_metrics_exp


def cell_to_image(region, pad=4):
    """Turn one cell's binary silhouette into a square RGB PIL image."""
    sil = np.pad(region.image, pad).astype(np.uint8) * 255      # cell = white, background = black
    h, w = sil.shape
    side = max(h, w)
    square = np.zeros((side, side), dtype=np.uint8)              # pad to square so resizing doesn't distort the shape
    square[(side - h)//2:(side - h)//2 + h, (side - w)//2:(side - w)//2 + w] = sil
    return Image.fromarray(square).convert("RGB")                # ViT expects 3 channels

@torch.no_grad()
def embed_mask(mask, batch_size=32):

#Embedding scripts
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224")
    model = ViTModel.from_pretrained("google/vit-base-patch16-224", add_pooling_layer=False).eval().to(device)

    regions = measure.regionprops(mask)                          # one region per cell label (0 is skipped automatically)
    labels = [r.label for r in regions]
    embeddings = []
    for i in range(0, len(regions), batch_size):
        batch = [cell_to_image(r) for r in regions[i:i + batch_size]]
        inputs = processor(images=batch, return_tensors="pt").to(device)
        out = model(**inputs).last_hidden_state                  # (batch, 197, 768)
        embeddings.append(out[:, 1:, :].mean(dim=1).cpu().numpy())  # mean of patch tokens; out[:, 0] would be the CLS token
    emb = np.vstack(embeddings)
    return pd.DataFrame(emb, index=labels, columns=[f"vit_{j}" for j in range(emb.shape[1])])

def pca_embeddings(embeddings):
    """Perform PCA on the embeddings and return a DataFrame with the principal components."""
    X = StandardScaler().fit_transform(embeddings)   # embeddings have arbitrary scale
    for n in [5,10, 20, 50]:
        pca = PCA(n_components=n)
        pcs = pca.fit_transform(X)
        # store as embeddings_pca_{n}
        embeddings_pca = pd.DataFrame(pcs, index=embeddings.index, columns=[f"vit_pca_{j}" for j in range(n)])
    return embeddings_pca


