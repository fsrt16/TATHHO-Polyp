import os
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt


def create_mapping_csv(image_dir, mask_dir, output_csv):
    """
    Maps images to masks using filename match and saves to CSV.

    Args:
        image_dir (str): Path to image directory.
        mask_dir (str): Path to mask directory.
        output_csv (str): Path to output CSV file.

    Returns:
        pd.DataFrame: DataFrame with image-mask pairs.
    """
    image_files = sorted(os.listdir(image_dir))
    mask_files = sorted(os.listdir(mask_dir))
    
    # Match filenames directly
    common_files = set(image_files).intersection(set(mask_files))
    mapped_data = [{"Image": file, "Mask": file} for file in common_files]
    
    df = pd.DataFrame(mapped_data)
    df.to_csv(output_csv, index=False)
    print(f"[INFO] Mapping CSV created at: {output_csv}")
    return df


def load_dataset(df, image_dir, mask_dir, size=(128, 128)):
    """
    Loads and resizes images and masks from file paths in the dataframe.

    Args:
        df (pd.DataFrame): DataFrame with Image and Mask columns.
        image_dir (str): Path to images.
        mask_dir (str): Path to masks.
        size (tuple): Resize dimensions.

    Returns:
        Tuple[np.ndarray, np.ndarray]: Arrays for images and binary masks.
    """
    Xdata, Ydata = [], []

    for _, row in tqdm(df.iterrows(), total=len(df)):
        # Load and preprocess image
        img_path = os.path.join(image_dir, row['Image'])
        img = Image.open(img_path).resize(size).convert('L')
        img_array = np.array(img)

        # Load and preprocess mask
        mask_path = os.path.join(mask_dir, row['Mask'])
        mask = Image.open(mask_path).resize(size)
        mask_array = np.array(mask)

        Xdata.append(img_array)
        Ydata.append(mask_array)

    Xdata = np.array(Xdata)
    Ydata = np.array(Ydata)

    # Convert all non-zero pixels in mask to 1
    Ydata_bin = np.where(Ydata > 0, 1, 0)

    print(f"[INFO] Xdata shape: {Xdata.shape}")
    print(f"[INFO] Ydata shape (binary): {Ydata_bin.shape}")
    return Xdata, Ydata_bin


def visualize_sample(index, X, Y):
    """
    Plots the image, mask, and overlap for a given index.

    Args:
        index (int): Index of the sample.
        X (np.ndarray): Image array.
        Y (np.ndarray): Mask array.
    """
    plt.figure(figsize=(18, 6))
    plt.subplot(1, 3, 1)
    plt.imshow(X[index], cmap='gray')
    plt.title('Input Image')

    plt.subplot(1, 3, 2)
    plt.imshow(Y[index], cmap='gray')
    plt.title('Mask')

    plt.subplot(1, 3, 3)
    plt.imshow(X[index], cmap='gray')
    plt.imshow(Y[index], cmap='hot', alpha=0.5)
    plt.title('Overlap')
    plt.show()


def plot_samples_with_masks(X_data, Y_data, num_samples=5):
    """
    Visualizes samples with visible masks.

    Args:
        X_data (np.ndarray): Images.
        Y_data (np.ndarray): Corresponding binary masks.
        num_samples (int): Number of samples to display.
    """
    count = 0
    for i in range(len(Y_data)):
        if np.sum(Y_data[i]) > 0:
            visualize_sample(i, X_data, Y_data)
            count += 1
        if count >= num_samples:
            break


# Example usage (Uncomment when running as standalone)
# base_dir = '/kaggle/input/kvasirseg/Kvasir-SEG/Kvasir-SEG/'
# image_dir = os.path.join(base_dir, 'images')
# mask_dir = os.path.join(base_dir, 'masks')
# output_csv = 'image_mask_mapping.csv'
# df = create_mapping_csv(image_dir, mask_dir, output_csv)
# Xdata, Ydata = load_dataset(df, image_dir, mask_dir)
# plot_samples_with_masks(Xdata, Ydata)
