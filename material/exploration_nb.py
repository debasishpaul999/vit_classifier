{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# CIFAR-10 Data Exploration\n",
    "\n",
    "This notebook explores the CIFAR-10 dataset and visualizes data augmentations."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "import sys\n",
    "sys.path.append('../src')\n",
    "\n",
    "import torch\n",
    "import torchvision\n",
    "import numpy as np\n",
    "import matplotlib.pyplot as plt\n",
    "import seaborn as sns\n",
    "from collections import Counter\n",
    "\n",
    "from data import get_dataloaders, get_class_names, denormalize\n",
    "from utils import load_config\n",
    "\n",
    "%matplotlib inline\n",
    "plt.style.use('seaborn-v0_8-darkgrid')"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 1. Load Configuration and Data"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Load config\n",
    "config = load_config('../config.yaml')\n",
    "print(\"Configuration loaded successfully!\")\n",
    "\n",
    "# Load data\n",
    "train_loader, val_loader, test_loader = get_dataloaders(config)\n",
    "class_names = get_class_names()\n",
    "\n",
    "print(f\"\\nClass names: {class_names}\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 2. Dataset Statistics"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Count samples\n",
    "print(f\"Training batches: {len(train_loader)}\")\n",
    "print(f\"Validation batches: {len(val_loader)}\")\n",
    "print(f\"Test batches: {len(test_loader)}\")\n",
    "\n",
    "# Get a batch\n",
    "images, labels = next(iter(train_loader))\n",
    "print(f\"\\nBatch shape: {images.shape}\")\n",
    "print(f\"Labels shape: {labels.shape}\")\n",
    "print(f\"Image range: [{images.min():.2f}, {images.max():.2f}]\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 3. Class Distribution"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Collect all labels\n",
    "all_labels = []\n",
    "for _, labels in train_loader:\n",
    "    all_labels.extend(labels.numpy())\n",
    "\n",
    "label_counts = Counter(all_labels)\n",
    "\n",
    "# Plot distribution\n",
    "plt.figure(figsize=(12, 6))\n",
    "plt.bar(range(10), [label_counts[i] for i in range(10)])\n",
    "plt.xticks(range(10), class_names, rotation=45)\n",
    "plt.xlabel('Class')\n",
    "plt.ylabel('Number of Samples')\n",
    "plt.title('Training Set Class Distribution')\n",
    "plt.grid(axis='y', alpha=0.3)\n",
    "plt.tight_layout()\n",
    "plt.show()\n",
    "\n",
    "print(\"\\nClass distribution:\")\n",
    "for i, name in enumerate(class_names):\n",
    "    print(f\"{name:12s}: {label_counts[i]:5d} samples\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 4. Sample Images"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Visualize random samples\n",
    "fig, axes = plt.subplots(4, 8, figsize=(16, 8))\n",
    "\n",
    "images, labels = next(iter(train_loader))\n",
    "\n",
    "for idx, ax in enumerate(axes.flat):\n",
    "    if idx < len(images):\n",
    "        # Denormalize\n",
    "        img = denormalize(\n",
    "            images[idx],\n",
    "            config['augmentation']['normalize_mean'],\n",
    "            config['augmentation']['normalize_std']\n",
    "        )\n",
    "        img = torch.clamp(img, 0, 1)\n",
    "        \n",
    "        ax.imshow(img.permute(1, 2, 0))\n",
    "        ax.set_title(class_names[labels[idx]], fontsize=10)\n",
    "        ax.axis('off')\n",
    "\n",
    "plt.suptitle('Sample Training Images with Augmentation', fontsize=16, fontweight='bold')\n",
    "plt.tight_layout()\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 5. Per-Class Examples"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Show examples for each class\n",
    "fig, axes = plt.subplots(10, 8, figsize=(16, 20))\n",
    "\n",
    "# Collect images per class\n",
    "class_images = {i: [] for i in range(10)}\n",
    "\n",
    "for images, labels in train_loader:\n",
    "    for img, label in zip(images, labels):\n",
    "        label = label.item()\n",
    "        if len(class_images[label]) < 8:\n",
    "            class_images[label].append(img)\n",
    "    \n",
    "    # Break if we have enough\n",
    "    if all(len(v) >= 8 for v in class_images.values()):\n",
    "        break\n",
    "\n",
    "# Plot\n",
    "for class_idx in range(10):\n",
    "    for img_idx in range(8):\n",
    "        ax = axes[class_idx, img_idx]\n",
    "        \n",
    "        img = denormalize(\n",
    "            class_images[class_idx][img_idx],\n",
    "            config['augmentation']['normalize_mean'],\n",
    "            config['augmentation']['normalize_std']\n",
    "        )\n",
    "        img = torch.clamp(img, 0, 1)\n",
    "        \n",
    "        ax.imshow(img.permute(1, 2, 0))\n",
    "        ax.axis('off')\n",
    "        \n",
    "        if img_idx == 0:\n",
    "            ax.set_ylabel(class_names[class_idx], fontsize=12, fontweight='bold')\n",
    "\n",
    "plt.suptitle('Examples from Each Class', fontsize=16, fontweight='bold')\n",
    "plt.tight_layout()\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 6. Patch Visualization"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Visualize how images are split into patches\n",
    "from model import PatchExtractor\n",
    "\n",
    "patch_size = config['model']['patch_size']\n",
    "patch_extractor = PatchExtractor(patch_size)\n",
    "\n",
    "# Get one image\n",
    "img_tensor = images[0:1]\n",
    "patches = patch_extractor(img_tensor)\n",
    "\n",
    "print(f\"Image shape: {img_tensor.shape}\")\n",
    "print(f\"Patches shape: {patches.shape}\")\n",
    "print(f\"Number of patches: {patches.shape[1]}\")\n",
    "print(f\"Patch dimension: {patches.shape[2]}\")\n",
    "\n",
    "# Visualize original and patches\n",
    "fig = plt.figure(figsize=(15, 12))\n",
    "\n",
    "# Original image\n",
    "ax = plt.subplot(1, 2, 1)\n",
    "img_display = denormalize(\n",
    "    img_tensor[0],\n",
    "    config['augmentation']['normalize_mean'],\n",
    "    config['augmentation']['normalize_std']\n",
    ")\n",
    "img_display = torch.clamp(img_display, 0, 1)\n",
    "ax.imshow(img_display.permute(1, 2, 0))\n",
    "ax.set_title('Original Image (72×72)', fontsize=14, fontweight='bold')\n",
    "ax.axis('off')\n",
    "\n",
    "# Draw grid\n",
    "image_size = config['data']['image_size']\n",
    "for i in range(0, image_size, patch_size):\n",
    "    ax.axhline(i, color='red', linewidth=1, alpha=0.5)\n",
    "    ax.axvline(i, color='red', linewidth=1, alpha=0.5)\n",
    "\n",
    "# Patches grid\n",
    "ax = plt.subplot(1, 2, 2)\n",
    "num_patches_per_side = image_size // patch_size\n",
    "\n",
    "# Reconstruct patches into grid\n",
    "patches_reshaped = patches[0].reshape(num_patches_per_side, num_patches_per_side, 3, patch_size, patch_size)\n",
    "patches_reshaped = patches_reshaped.permute(0, 3, 1, 4, 2).contiguous()\n",
    "patches_grid = patches_reshaped.reshape(image_size, image_size, 3)\n",
    "\n",
    "# Denormalize\n",
    "mean = torch.tensor(config['augmentation']['normalize_mean']).view(1, 1, 3)\n",
    "std = torch.tensor(config['augmentation']['normalize_std']).view(1, 1, 3)\n",
    "patches_grid = patches_grid * std + mean\n",
    "patches_grid = torch.clamp(patches_grid, 0, 1)\n",
    "\n",
    "ax.imshow(patches_grid.numpy())\n",
    "ax.set_title(f'Extracted Patches ({num_patches_per_side}×{num_patches_per_side} = {patches.shape[1]} patches)',\n",
    "             fontsize=14, fontweight='bold')\n",
    "ax.axis('off')\n",
    "\n",
    "plt.tight_layout()\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 7. Data Augmentation Comparison"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Load same image with and without augmentation\n",
    "import torchvision.transforms as transforms\n",
    "\n",
    "# Load raw CIFAR-10\n",
    "cifar_raw = torchvision.datasets.CIFAR10(\n",
    "    root=config['data']['data_dir'],\n",
    "    train=True,\n",
    "    download=False,\n",
    "    transform=transforms.ToTensor()\n",
    ")\n",
    "\n",
    "# Get one image\n",
    "idx = np.random.randint(0, len(cifar_raw))\n",
    "raw_img, label = cifar_raw[idx]\n",
    "\n",
    "# Apply augmentations multiple times\n",
    "from data import get_transforms\n",
    "train_transform = get_transforms(config, train=True)\n",
    "\n",
    "fig, axes = plt.subplots(2, 5, figsize=(15, 6))\n",
    "\n",
    "# Original\n",
    "axes[0, 0].imshow(raw_img.permute(1, 2, 0))\n",
    "axes[0, 0].set_title('Original (32×32)', fontweight='bold')\n",
    "axes[0, 0].axis('off')\n",
    "\n",
    "# Augmented versions\n",
    "from PIL import Image\n",
    "raw_pil = Image.fromarray((raw_img.permute(1, 2, 0).numpy() * 255).astype(np.uint8))\n",
    "\n",
    "for i in range(1, 5):\n",
    "    aug_img = train_transform(raw_pil)\n",
    "    \n",
    "    # Denormalize\n",
    "    aug_display = denormalize(\n",
    "        aug_img,\n",
    "        config['augmentation']['normalize_mean'],\n",
    "        config['augmentation']['normalize_std']\n",
    "    )\n",
    "    aug_display = torch.clamp(aug_display, 0, 1)\n",
    "    \n",
    "    axes[0, i].imshow(aug_display.permute(1, 2, 0))\n",
    "    axes[0, i].set_title(f'Augmented {i}', fontweight='bold')\n",
    "    axes[0, i].axis('off')\n",
    "\n",
    "# More augmentations\n",
    "for i in range(5):\n",
    "    aug_img = train_transform(raw_pil)\n",
    "    \n",
    "    aug_display = denormalize(\n",
    "        aug_img,\n",
    "        config['augmentation']['normalize_mean'],\n",
    "        config['augmentation']['normalize_std']\n",
    "    )\n",
    "    aug_display = torch.clamp(aug_display, 0, 1)\n",
    "    \n",
    "    axes[1, i].imshow(aug_display.permute(1, 2, 0))\n",
    "    axes[1, i].set_title(f'Augmented {i+5}', fontweight='bold')\n",
    "    axes[1, i].axis('off')\n",
    "\n",
    "plt.suptitle(f'Data Augmentation Examples - Class: {class_names[label]}',\n",
    "             fontsize=16, fontweight='bold')\n",
    "plt.tight_layout()\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 8. Summary Statistics"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "print(\"=\"*60)\n",
    "print(\"CIFAR-10 Dataset Summary\")\n",
    "print(\"=\"*60)\n",
    "print(f\"Number of classes: {len(class_names)}\")\n",
    "print(f\"Class names: {', '.join(class_names)}\")\n",
    "print(f\"\\nOriginal image size: 32×32×3\")\n",
    "print(f\"Resized image size: {config['data']['image_size']}×{config['data']['image_size']}×3\")\n",
    "print(f\"Patch size: {patch_size}×{patch_size}\")\n",
    "print(f\"Number of patches: {(config['data']['image_size']//patch_size)**2}\")\n",
    "print(f\"\\nBatch size: {config['data']['batch_size']}\")\n",
    "print(f\"Training samples: {len(train_loader.dataset)}\")\n",
    "print(f\"Validation samples: {len(val_loader.dataset)}\")\n",
    "print(f\"Test samples: {len(test_loader.dataset)}\")\n",
    "print(\"=\"*60)"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.8.0"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}