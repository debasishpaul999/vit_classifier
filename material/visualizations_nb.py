{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Vision Transformer Results & Attention Visualization\n",
    "\n",
    "This notebook visualizes model predictions, attention maps, and analyzes results."
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
    "import numpy as np\n",
    "import matplotlib.pyplot as plt\n",
    "import seaborn as sns\n",
    "from sklearn.metrics import confusion_matrix, classification_report\n",
    "\n",
    "from model import create_model\n",
    "from data import get_dataloaders, get_class_names, denormalize\n",
    "from utils import load_config, load_checkpoint\n",
    "\n",
    "%matplotlib inline\n",
    "plt.style.use('seaborn-v0_8-darkgrid')"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 1. Load Model and Data"
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
    "device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')\n",
    "print(f\"Using device: {device}\")\n",
    "\n",
    "# Load data\n",
    "train_loader, val_loader, test_loader = get_dataloaders(config)\n",
    "class_names = get_class_names()\n",
    "\n",
    "# Create and load model\n",
    "model = create_model(config).to(device)\n",
    "checkpoint_path = '../results/checkpoints/checkpoint_epoch_1_best.pth'\n",
    "load_checkpoint(checkpoint_path, model)\n",
    "model.eval()\n",
    "\n",
    "print(\"Model loaded successfully!\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 2. Model Predictions"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Get predictions on test set\n",
    "all_preds = []\n",
    "all_labels = []\n",
    "all_probs = []\n",
    "\n",
    "with torch.no_grad():\n",
    "    for images, labels in test_loader:\n",
    "        images = images.to(device)\n",
    "        outputs = model(images)\n",
    "        probs = torch.softmax(outputs, dim=1)\n",
    "        _, preds = torch.max(outputs, 1)\n",
    "        \n",
    "        all_preds.extend(preds.cpu().numpy())\n",
    "        all_labels.extend(labels.numpy())\n",
    "        all_probs.extend(probs.cpu().numpy())\n",
    "\n",
    "all_preds = np.array(all_preds)\n",
    "all_labels = np.array(all_labels)\n",
    "all_probs = np.array(all_probs)\n",
    "\n",
    "# Calculate accuracy\n",