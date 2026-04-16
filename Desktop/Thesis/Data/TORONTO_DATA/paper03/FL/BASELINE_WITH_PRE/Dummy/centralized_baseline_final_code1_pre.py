# ===========================================================
# scenario1_baseline_centralized.py
# Scenario 1: Baseline Centralized Model with 5-Fold CV
# COMPLETE FIXED CODE - NO KEY ERRORS
# ===========================================================

import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
from pathlib import Path
from torchvision import models
from datetime import datetime
import warnings
from scipy import stats

warnings.filterwarnings('ignore')

# ================== CONFIG ==================
MANIFEST = r"C:\Users\Nicola Sambo\Desktop\Thesis\Data\TORONTO_DATA\paper03\manifest.csv"
OUT_DIR = r"C:\Users\Nicola Sambo\Desktop\Thesis\Data\TORONTO_DATA\paper03\FL\BASELINE_WITH_PRE\RESULTS"
BATCH_SIZE = 4
IMG_SIZE = 224
T = 16
LR = 1e-4
EPOCHS = 50
NUM_CLASSES = 2
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Create output directory
Path(OUT_DIR).mkdir(parents=True, exist_ok=True)
print(f"🚀 SCENARIO 1: Baseline Centralized Model")
print(f"📁 Output directory: {OUT_DIR}")
print(f"💻 Device: {DEVICE}")
# ============================================

# ================== MEDIAPIPE FACE DETECTION ==================
import mediapipe as mp

class MediaPipeFaceDetector:
    def __init__(self):
        self.mp_face_detection = mp.solutions.face_detection
        self.face_detection = self.mp_face_detection.FaceDetection(
            model_selection=1,  # 1 for full-range detection
            min_detection_confidence=0.5
        )
    
    def detect_and_crop_face(self, image):
        """Detect face and crop around it"""
        try:
            # Convert PIL to RGB numpy array
            rgb_image = np.array(image)
            
            # Detect faces
            results = self.face_detection.process(rgb_image)
            
            if results.detections:
                # Get first face detection
                detection = results.detections[0]
                bbox = detection.location_data.relative_bounding_box
                
                h, w = rgb_image.shape[:2]
                
                # Calculate bounding box coordinates
                x = int(bbox.xmin * w)
                y = int(bbox.ymin * h)
                width = int(bbox.width * w)
                height = int(bbox.height * h)
                
                # Add some padding (20% of face size)
                padding_x = int(width * 0.5)
                padding_y = int(height * 0.5)
                
                # Ensure coordinates are within image bounds
                x1 = max(0, x - padding_x)
                y1 = max(0, y - padding_y)
                x2 = min(w, x + width + padding_x)
                y2 = min(h, y + height + padding_y)
                
                # Crop face region
                face_crop = rgb_image[y1:y2, x1:x2]
                
                # Convert back to PIL
                return Image.fromarray(face_crop)
            else:
                # No face detected, return original image
                return image
                
        except Exception as e:
            # If any error, return original image
            return image
    
    def __del__(self):
        if hasattr(self, 'face_detection'):
            self.face_detection.close()
# ============================================

# ================== MODEL ARCHITECTURE ==================
class ResNet18FeatureExtractor(nn.Module):
    def __init__(self, dropout_rate=0.3):
        super().__init__()
        base = models.resnet18(weights=None)
        self.trunk = nn.Sequential(*list(base.children())[:-2])
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(dropout_rate)
        for p in self.trunk.parameters(): 
            p.requires_grad = True

    def forward(self, x):
        h = self.trunk(x)
        h = self.gap(h).flatten(1)
        h = self.dropout(h)
        return h

class BiGRUHead(nn.Module):
    def __init__(self, in_dim=512, hidden=128, num_classes=2):
        super().__init__()
        self.gru = nn.GRU(in_dim, hidden, batch_first=True, bidirectional=True)
        self.drop1 = nn.Dropout(0.3)
        self.fc1 = nn.Linear(2 * hidden, hidden // 2)
        self.drop2 = nn.Dropout(0.3)
        self.fc2 = nn.Linear(hidden // 2, num_classes)
        self.relu = nn.ReLU()

    def forward(self, seq):
        out, _ = self.gru(seq)
        last = out[:, -1]
        x = self.drop1(last)
        x = self.relu(self.fc1(x))
        x = self.drop2(x)
        return self.fc2(x)

class ResNet18_BiGRU(nn.Module):
    def __init__(self, hidden=128, num_classes=2):
        super().__init__()
        self.feats = ResNet18FeatureExtractor(dropout_rate=0.3)
        self.head = BiGRUHead(512, hidden, num_classes)

    def forward(self, x):
        B, T, C, H, W = x.shape
        f = self.feats(x.view(B * T, C, H, W)).view(B, T, 512)
        return self.head(f)
# ============================================

# ================== DATASET ==================
def list_images_sorted(folder):
    exts = (".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG")
    try:
        names = [f for f in sorted(os.listdir(folder)) if f.endswith(exts)]
    except FileNotFoundError:
        names = []
    return [str(Path(folder) / f) for f in names]

def sample_indices_tsn(num_frames, T, jitter=True):
    if num_frames <= 0:
        return []
    bounds = np.linspace(0, num_frames, T + 1, dtype=int)
    idx = []
    for i in range(T):
        s, e = bounds[i], max(bounds[i + 1] - 1, bounds[i])
        if jitter and e > s:
            idx.append(np.random.randint(s, e + 1))
        else:
            idx.append((s + e) // 2)
    return idx

class SimpleVideoDataset(Dataset):
    def __init__(self, df, T=16, img_size=224, normalize=True, train_mode=True):
        self.df = df.reset_index(drop=True)
        self.T = T
        self.img_size = img_size
        self.normalize = normalize
        self.train_mode = train_mode
        
        # Initialize MediaPipe face detector
        self.face_detector = MediaPipeFaceDetector()
        
        # Fixed ImageNet normalization (BASELINE - no advanced preprocessing)
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        r = self.df.iloc[i]
        frames = list_images_sorted(r["clip_dir"])
        
        if len(frames) == 0:
            video = torch.zeros(self.T, 3, self.img_size, self.img_size)
            return video, torch.tensor(r["label"], dtype=torch.long)

        idxs = sample_indices_tsn(len(frames), self.T, jitter=self.train_mode)
        imgs = []
        
        for j in idxs:
            p = frames[min(j, len(frames) - 1)]
            try:
                from PIL import Image
                with Image.open(p) as im:
                    # MEDIAPIPE FACE DETECTION + CROPPING
                    face_cropped_im = self.face_detector.detect_and_crop_face(im)
                    
                    # Resize to required input size for ResNet-BiGRU
                    xpil = face_cropped_im.resize((self.img_size, self.img_size))
                    
                    # BASIC PREPROCESSING: Only resize + normalization
                    x = torch.tensor(np.array(xpil), dtype=torch.float32).permute(2, 0, 1) / 255.0
                    if self.normalize:
                        x = (x - self.mean) / self.std
            except Exception:
                x = torch.zeros(3, self.img_size, self.img_size)
            imgs.append(x)

        video = torch.stack(imgs, dim=0)
        label = int(r["label"])
        return video, torch.tensor(label, dtype=torch.long)
# ============================================

# ================== TRAINING FUNCTIONS ==================
def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    loss_sum, n, correct = 0.0, 0, 0
    for X, y in loader:
        X, y = X.to(device), y.to(device)
        optimizer.zero_grad()
        out = model(X)
        loss = criterion(out, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        loss_sum += loss.item() * X.size(0)
        correct += (out.argmax(1) == y).sum().item()
        n += X.size(0)
    return loss_sum / max(n, 1), correct / max(n, 1)

def evaluate_comprehensive(model, loader, criterion, device):
    """Comprehensive evaluation with all metrics"""
    model.eval()
    loss_sum, n = 0.0, 0
    all_true, all_pred, all_probs = [], [], []
    
    with torch.no_grad():
        for X, y in loader:
            X, y = X.to(device), y.to(device)
            out = model(X)
            loss = criterion(out, y)
            loss_sum += loss.item() * X.size(0)
            n += X.size(0)
            
            probs = torch.softmax(out, dim=1)[:, 1]
            preds = out.argmax(dim=1)
            
            all_true.extend(y.cpu().numpy())
            all_pred.extend(preds.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    
    # Calculate metrics
    accuracy = accuracy_score(all_true, all_pred)
    f1 = f1_score(all_true, all_pred, zero_division=0)
    precision = precision_score(all_true, all_pred, zero_division=0)
    recall = recall_score(all_true, all_pred, zero_division=0)
    
    # Specificity
    cm = confusion_matrix(all_true, all_pred, labels=[0, 1])
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    else:
        specificity = 0
    
    # AUC
    if len(np.unique(all_true)) == 2:
        auc = roc_auc_score(all_true, all_probs)
    else:
        auc = 0
    
    metrics = {
        'loss': loss_sum / max(n, 1),
        'accuracy': accuracy, 'f1': f1, 'precision': precision, 
        'recall': recall, 'specificity': specificity, 'auc': auc
    }
    
    return metrics, np.array(all_true), np.array(all_probs)
# ============================================

# ================== GRAPH FUNCTIONS ==================
def plot_fold_loss_curve(fold_num, train_losses, val_losses, save_dir):
    """Plot ONLY loss curve for a single fold (separate from accuracy)"""
    epochs = range(1, len(train_losses) + 1)
    
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_losses, 'b-', label='Training Loss', linewidth=2)
    plt.plot(epochs, val_losses, 'r-', label='Validation Loss', linewidth=2)
    plt.title(f'Fold {fold_num} - Training & Validation Loss', fontsize=14, fontweight='bold')
    plt.xlabel('Epochs', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    save_path = os.path.join(save_dir, f"fold_{fold_num}_loss_curve.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"   ✅ Saved: {os.path.basename(save_path)}")
    return save_path

def plot_fold_accuracy_curve(fold_num, train_accs, val_accs, save_dir):
    """Plot ONLY accuracy curve for a single fold (separate from loss)"""
    epochs = range(1, len(train_accs) + 1)
    
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_accs, 'b-', label='Training Accuracy', linewidth=2)
    plt.plot(epochs, val_accs, 'r-', label='Validation Accuracy', linewidth=2)
    plt.title(f'Fold {fold_num} - Training & Validation Accuracy', fontsize=14, fontweight='bold')
    plt.xlabel('Epochs', fontsize=12)
    plt.ylabel('Accuracy', fontsize=12)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.ylim(0, 1.0)
    plt.tight_layout()
    
    save_path = os.path.join(save_dir, f"fold_{fold_num}_accuracy_curve.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"   ✅ Saved: {os.path.basename(save_path)}")
    return save_path

def plot_fold_roc_curve(fold_num, y_true, y_probs, save_dir):
    """Plot ROC curve for a single fold"""
    fpr, tpr, _ = roc_curve(y_true, y_probs)
    auc_score = roc_auc_score(y_true, y_probs)
    
    plt.figure(figsize=(8, 8))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {auc_score:.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=1, linestyle='--', alpha=0.5, label='Chance')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title(f'Fold {fold_num} - ROC Curve', fontsize=14, fontweight='bold')
    plt.legend(loc="lower right", fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    save_path = os.path.join(save_dir, f"fold_{fold_num}_roc_curve.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"   ✅ Saved: {os.path.basename(save_path)}")
    return save_path

def plot_fold_metrics(fold_num, test_metrics, save_dir):
    """Plot metrics bar chart for a single fold"""
    metrics_names = ['accuracy', 'f1', 'precision', 'recall', 'specificity', 'auc']
    values = [test_metrics[name] for name in metrics_names]
    
    plt.figure(figsize=(12, 6))
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
    bars = plt.bar(metrics_names, values, color=colors, alpha=0.8)
    
    plt.ylim(0, 1.0)
    plt.ylabel('Score', fontsize=12)
    plt.title(f'Fold {fold_num} - Test Set Performance Metrics', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar, value in zip(bars, values):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                f'{value:.3f}', ha='center', va='bottom', fontweight='bold', fontsize=11)
    
    save_path = os.path.join(save_dir, f"fold_{fold_num}_metrics.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"   ✅ Saved: {os.path.basename(save_path)}")
    return save_path

def plot_mean_roc_curve(all_fold_data, save_dir):
    """Plot mean ROC curve with standard deviation - ESSENTIAL for papers"""
    plt.figure(figsize=(8, 8))
    
    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)
    
    for fold_data in all_fold_data:
        fpr, tpr, _ = roc_curve(fold_data['true_labels'], fold_data['probs'])
        roc_auc = roc_auc_score(fold_data['true_labels'], fold_data['probs'])
        
        # Interpolate to common FPR points
        interp_tpr = np.interp(mean_fpr, fpr, tpr)
        interp_tpr[0] = 0.0
        tprs.append(interp_tpr)
        aucs.append(roc_auc)
    
    # Calculate mean and std
    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    std_tpr = np.std(tprs, axis=0)
    
    mean_auc = np.mean(aucs)
    std_auc = np.std(aucs)
    
    # Plot mean ROC
    plt.plot(mean_fpr, mean_tpr, color='b', 
             label=f'Mean ROC (AUC = {mean_auc:.3f} ± {std_auc:.3f})', lw=2)
    
    # Plot std deviation
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=0.3,
                    label=f'±1 std dev')
    
    # Plot chance line
    plt.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Chance')
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('Mean ROC Curve with Standard Deviation\n(5-Fold Cross-Validation)', 
              fontsize=14, fontweight='bold')
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    
    save_path = os.path.join(save_dir, "scenario1_mean_roc_curve.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"   ✅ Saved: {os.path.basename(save_path)}")
    return save_path

def plot_metrics_distribution(all_fold_metrics, save_dir):
    """Box plot showing distribution of metrics across folds - STANDARD in papers"""
    metrics = ['accuracy', 'f1', 'precision', 'recall', 'specificity', 'auc']
    data = {metric: [fold[metric] for fold in all_fold_metrics] for metric in metrics}
    
    fig, ax = plt.subplots(figsize=(12, 6))
    boxes = ax.boxplot([data[metric] for metric in metrics], 
                       labels=metrics, patch_artist=True)
    
    # Customize colors
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
    for patch, color in zip(boxes['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('Performance Metrics Distribution Across 5 Folds', 
                 fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim(0, 1.0)
    
    # Add mean values as text
    for i, metric in enumerate(metrics):
        mean_val = np.mean(data[metric])
        ax.text(i + 1, 0.05, f'μ={mean_val:.3f}', 
                ha='center', va='bottom', fontweight='bold', fontsize=10)
    
    save_path = os.path.join(save_dir, "scenario1_metrics_distribution.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"   ✅ Saved: {os.path.basename(save_path)}")
    return save_path

def plot_training_convergence_separate(all_fold_histories, save_dir):
    """Plot average training convergence with SEPARATE graphs for loss and accuracy"""
    
    # Find maximum epochs across all folds
    max_epochs = max(len(hist['train_acc']) for hist in all_fold_histories)
    
    # Prepare arrays for mean curves
    all_train_acc = np.full((len(all_fold_histories), max_epochs), np.nan)
    all_val_acc = np.full((len(all_fold_histories), max_epochs), np.nan)
    all_train_loss = np.full((len(all_fold_histories), max_epochs), np.nan)
    all_val_loss = np.full((len(all_fold_histories), max_epochs), np.nan)
    
    # Fill arrays
    for i, history in enumerate(all_fold_histories):
        epochs = len(history['train_acc'])
        all_train_acc[i, :epochs] = history['train_acc']
        all_val_acc[i, :epochs] = history['val_acc']
        all_train_loss[i, :epochs] = history['train_loss']
        all_val_loss[i, :epochs] = history['val_loss']
    
    # Calculate mean and std
    mean_train_acc = np.nanmean(all_train_acc, axis=0)
    mean_val_acc = np.nanmean(all_val_acc, axis=0)
    std_train_acc = np.nanstd(all_train_acc, axis=0)
    std_val_acc = np.nanstd(all_val_acc, axis=0)
    
    mean_train_loss = np.nanmean(all_train_loss, axis=0)
    mean_val_loss = np.nanmean(all_val_loss, axis=0)
    std_train_loss = np.nanstd(all_train_loss, axis=0)
    std_val_loss = np.nanstd(all_val_loss, axis=0)
    
    epochs_range = range(1, max_epochs + 1)
    
    # Create SEPARATE figures for accuracy and loss
    
    # Figure 1: Accuracy only
    plt.figure(figsize=(10, 6))
    plt.plot(epochs_range, mean_train_acc, 'b-', linewidth=3, label='Mean Training Accuracy')
    plt.plot(epochs_range, mean_val_acc, 'r-', linewidth=3, label='Mean Validation Accuracy')
    plt.fill_between(epochs_range, mean_train_acc - std_train_acc, mean_train_acc + std_train_acc, 
                    alpha=0.2, color='blue', label='Training ±1 std')
    plt.fill_between(epochs_range, mean_val_acc - std_val_acc, mean_val_acc + std_val_acc, 
                    alpha=0.2, color='red', label='Validation ±1 std')
    
    plt.xlabel('Epochs', fontsize=12)
    plt.ylabel('Accuracy', fontsize=12)
    plt.title('Average Accuracy Convergence\nAcross 5-Fold Cross-Validation', 
              fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.ylim(0, 1.0)
    
    accuracy_path = os.path.join(save_dir, "scenario1_accuracy_convergence.png")
    plt.savefig(accuracy_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"   ✅ Saved: {os.path.basename(accuracy_path)}")
    
    # Figure 2: Loss only
    plt.figure(figsize=(10, 6))
    plt.plot(epochs_range, mean_train_loss, 'b-', linewidth=3, label='Mean Training Loss')
    plt.plot(epochs_range, mean_val_loss, 'r-', linewidth=3, label='Mean Validation Loss')
    plt.fill_between(epochs_range, mean_train_loss - std_train_loss, mean_train_loss + std_train_loss, 
                    alpha=0.2, color='blue', label='Training ±1 std')
    plt.fill_between(epochs_range, mean_val_loss - std_val_loss, mean_val_loss + std_val_loss, 
                    alpha=0.2, color='red', label='Validation ±1 std')
    
    plt.xlabel('Epochs', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.title('Average Loss Convergence\nAcross 5-Fold Cross-Validation', 
              fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    
    loss_path = os.path.join(save_dir, "scenario1_loss_convergence.png")
    plt.savefig(loss_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"   ✅ Saved: {os.path.basename(loss_path)}")
    
    return accuracy_path, loss_path

def plot_training_stability(all_fold_histories, save_dir):
    """Plot all folds training curves to show stability"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot all folds
    for i, history in enumerate(all_fold_histories):
        epochs = range(1, len(history['train_acc']) + 1)
        ax1.plot(epochs, history['train_acc'], 'b-', alpha=0.3, linewidth=1)
        ax1.plot(epochs, history['val_acc'], 'r-', alpha=0.3, linewidth=1)
        ax2.plot(epochs, history['train_loss'], 'b-', alpha=0.3, linewidth=1)
        ax2.plot(epochs, history['val_loss'], 'r-', alpha=0.3, linewidth=1)
    
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Accuracy')
    ax1.set_title('Training Stability - Accuracy\n(All 5 Folds)')
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, 1.0)
    
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Loss')
    ax2.set_title('Training Stability - Loss\n(All 5 Folds)')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    save_path = os.path.join(save_dir, "scenario1_training_stability.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"   ✅ Saved: {os.path.basename(save_path)}")
    return save_path

def calculate_confidence_intervals(metrics_data, confidence=0.95):
    """Calculate 95% confidence intervals for metrics"""
    results = {}
    for metric in ['accuracy', 'f1', 'precision', 'recall', 'specificity', 'auc']:
        values = [fold[metric] for fold in metrics_data]
        n = len(values)
        mean = np.mean(values)
        std = np.std(values)
        
        # Calculate confidence interval
        se = std / np.sqrt(n)
        ci = stats.t.interval(confidence, n-1, loc=mean, scale=se)
        
        results[metric] = {
            'mean': mean,
            'std': std,
            'ci_lower': ci[0],
            'ci_upper': ci[1],
            'ci_width': ci[1] - ci[0]
        }
    
    return results
# ============================================

# ================== FIXED CSV SAVING FUNCTIONS ==================
def save_fold_training_history(fold_num, train_losses, val_losses, train_accs, val_accs, save_dir):
    """Save training history for a single fold"""
    history_df = pd.DataFrame({
        'epoch': range(1, len(train_losses) + 1),
        'train_loss': train_losses,
        'val_loss': val_losses,
        'train_accuracy': train_accs,
        'val_accuracy': val_accs
    })
    save_path = os.path.join(save_dir, f"fold_{fold_num}_training_history.csv")
    history_df.to_csv(save_path, index=False)
    print(f"   ✅ Saved: {os.path.basename(save_path)}")
    return save_path

def save_fold_predictions(fold_num, true_labels, predicted_probs, save_dir):
    """Save predictions for a single fold"""
    pred_df = pd.DataFrame({
        'true_label': true_labels,
        'predicted_prob': predicted_probs,
        'predicted_label': (predicted_probs > 0.5).astype(int)
    })
    save_path = os.path.join(save_dir, f"fold_{fold_num}_predictions.csv")
    pred_df.to_csv(save_path, index=False)
    print(f"   ✅ Saved: {os.path.basename(save_path)}")
    return save_path

def save_fold_confusion_matrix(fold_num, y_true, y_probs, save_dir):
    """Save confusion matrix for a single fold"""
    y_pred = (y_probs > 0.5).astype(int)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    
    cm_df = pd.DataFrame(cm, 
                        index=['Actual Healthy', 'Actual Stroke'],
                        columns=['Predicted Healthy', 'Predicted Stroke'])
    save_path = os.path.join(save_dir, f"fold_{fold_num}_confusion_matrix.csv")
    cm_df.to_csv(save_path)
    print(f"   ✅ Saved: {os.path.basename(save_path)}")
    return save_path

def save_all_folds_metrics_combined(all_fold_metrics, save_dir):
    """Save ALL fold metrics in a SINGLE CSV file - FIXED VERSION"""
    combined_data = []
    for fold_metrics in all_fold_metrics:
        combined_data.append({
            'fold': fold_metrics['fold'],
            'scenario': '1_baseline_centralized',
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'train_samples': fold_metrics['train_samples'],
            'test_samples': fold_metrics['test_samples'],
            # FIXED: Use the correct keys that exist in the dictionary
            'test_accuracy': fold_metrics['accuracy'],
            'test_f1': fold_metrics['f1'],
            'test_precision': fold_metrics['precision'],
            'test_recall': fold_metrics['recall'],
            'test_specificity': fold_metrics['specificity'],
            'test_auc': fold_metrics['auc'],
            'test_loss': fold_metrics['loss']
        })
    
    combined_df = pd.DataFrame(combined_data)
    save_path = os.path.join(save_dir, "all_folds_metrics_combined.csv")
    combined_df.to_csv(save_path, index=False)
    print(f"   ✅ Saved: {os.path.basename(save_path)}")
    return save_path

def save_overall_results(all_fold_metrics, confidence_intervals, save_dir):
    """Save aggregated results across all folds - FIXED VERSION"""
    # Create summary dataframe with correct key mapping
    summary_data = []
    for fold_metrics in all_fold_metrics:
        summary_data.append({
            'fold': fold_metrics['fold'],
            'test_accuracy': fold_metrics['accuracy'],
            'test_f1': fold_metrics['f1'],
            'test_precision': fold_metrics['precision'],
            'test_recall': fold_metrics['recall'],
            'test_specificity': fold_metrics['specificity'],
            'test_auc': fold_metrics['auc'],
            'test_loss': fold_metrics['loss'],
            'train_samples': fold_metrics['train_samples'],
            'test_samples': fold_metrics['test_samples']
        })
    
    summary_df = pd.DataFrame(summary_data)
    
    # Calculate mean and std across folds
    metrics_to_aggregate = ['test_accuracy', 'test_f1', 'test_precision', 'test_recall', 'test_specificity', 'test_auc']
    summary_stats = []
    
    for metric in metrics_to_aggregate:
        values = summary_df[metric]
        # Map back to original metric names for confidence intervals
        original_metric = metric.replace('test_', '')
        ci_data = confidence_intervals[original_metric]
        summary_stats.append({
            'metric': metric,
            'mean': np.mean(values),
            'std': np.std(values),
            'min': np.min(values),
            'max': np.max(values),
            'ci_lower': ci_data['ci_lower'],
            'ci_upper': ci_data['ci_upper'],
            'ci_width': ci_data['ci_width']
        })
    
    stats_df = pd.DataFrame(summary_stats)
    
    # Save both files
    summary_path = os.path.join(save_dir, "scenario1_all_folds_summary.csv")
    stats_path = os.path.join(save_dir, "scenario1_cross_validation_statistics.csv")
    
    summary_df.to_csv(summary_path, index=False)
    stats_df.to_csv(stats_path, index=False)
    
    print(f"   ✅ Saved: {os.path.basename(summary_path)}")
    print(f"   ✅ Saved: {os.path.basename(stats_path)}")
    
    return summary_df, stats_df
# ============================================

# ================== MAIN 5-FOLD CROSS VALIDATION ==================
def main():
    print(f"\n{'='*60}")
    print(f"🚀 SCENARIO 1: Baseline Centralized Model - 5-Fold CV")
    print(f"{'='*60}")
    print(f"🎯 Preprocessing: MEDIAPIPE FACE DETECTION + BASIC")
    print(f"📊 Model: ResNet18-BiGRU")
    print(f"💾 All results will be saved in: {OUT_DIR}")
    
    # Load data
    df = pd.read_csv(MANIFEST)
    subjects = df['subject_id'].unique()
    
    print(f"\n📁 Data Overview:")
    print(f"   Total samples: {len(df)}")
    print(f"   Total unique subjects: {len(subjects)}")
    print(f"   Classes: {df['label'].value_counts().to_dict()}")
    
    # Initialize k-fold
    kfold = KFold(n_splits=5, shuffle=True, random_state=42)
    
    all_fold_metrics = []
    all_fold_histories = []
    all_fold_data = []  # For ROC curves
    
    for fold, (train_val_idx, test_idx) in enumerate(kfold.split(subjects)):
        print(f"\n{'='*50}")
        print(f"🔄 PROCESSING FOLD {fold+1}/5")
        print(f"{'='*50}")
        
        # Get subjects for this fold
        train_val_subjects = subjects[train_val_idx]
        test_subjects = subjects[test_idx]
        
        # Further split train_val into train/validation (80/20 of the 80%)
        train_subjects, val_subjects = train_test_split(
            train_val_subjects, test_size=0.2, random_state=42
        )
        
        # Create dataframes
        train_df = df[df['subject_id'].isin(train_subjects)].reset_index(drop=True)
        val_df = df[df['subject_id'].isin(val_subjects)].reset_index(drop=True)
        test_df = df[df['subject_id'].isin(test_subjects)].reset_index(drop=True)
        
        print(f"📈 Data Distribution:")
        print(f"   Train: {len(train_df)} samples ({len(train_subjects)} subjects)")
        print(f"   Val:   {len(val_df)} samples ({len(val_subjects)} subjects)")
        print(f"   Test:  {len(test_df)} samples ({len(test_subjects)} subjects)")
        
        # Check class distribution
        print(f"🎯 Class Distribution:")
        for split_name, split_df in [('Train', train_df), ('Val', val_df), ('Test', test_df)]:
            healthy = len(split_df[split_df['label'] == 0])
            stroke = len(split_df[split_df['label'] == 1])
            print(f"   {split_name}: {healthy} healthy, {stroke} stroke")
        
        # Create data loaders
        train_loader = DataLoader(
            SimpleVideoDataset(train_df, T=T, img_size=IMG_SIZE, train_mode=True),
            batch_size=BATCH_SIZE, shuffle=True, num_workers=0
        )
        val_loader = DataLoader(
            SimpleVideoDataset(val_df, T=T, img_size=IMG_SIZE, train_mode=False),
            batch_size=BATCH_SIZE, shuffle=False, num_workers=0
        )
        test_loader = DataLoader(
            SimpleVideoDataset(test_df, T=T, img_size=IMG_SIZE, train_mode=False),
            batch_size=BATCH_SIZE, shuffle=False, num_workers=0
        )
        
        # Initialize model for this fold
        model = ResNet18_BiGRU(hidden=128, num_classes=NUM_CLASSES).to(DEVICE)
        optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-1)
        criterion = nn.CrossEntropyLoss()
        
        # Training history for this fold
        train_losses, val_losses = [], []
        train_accs, val_accs = [], []
        
        best_val_acc = 0
        best_model_state = None
        best_epoch = 0
        
        # Early stopping
        patience = 15
        patience_counter = 0
        
        print(f"📈 Training fold {fold+1}...")
        
        for epoch in range(EPOCHS):
            # Training
            train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, DEVICE)
            
            # Validation
            val_metrics, _, _ = evaluate_comprehensive(model, val_loader, criterion, DEVICE)
            val_loss, val_acc = val_metrics['loss'], val_metrics['accuracy']
            
            # Store history
            train_losses.append(train_loss)
            train_accs.append(train_acc)
            val_losses.append(val_loss)
            val_accs.append(val_acc)
            
            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_model_state = model.state_dict().copy()
                best_epoch = epoch + 1
                patience_counter = 0
            else:
                patience_counter += 1
            
            # Early stopping
            if patience_counter >= patience:
                print(f"   🛑 Early stopping at epoch {epoch+1}")
                break
            
            # Print progress
            if (epoch + 1) % 10 == 0 or epoch == 0:
                print(f"   Epoch {epoch+1:3d} | "
                      f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
                      f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")
        
        print(f"✅ Fold {fold+1} training completed!")
        print(f"   Best validation accuracy: {best_val_acc:.4f} at epoch {best_epoch}")
        
        # Load best model for testing
        model.load_state_dict(best_model_state)
        
        # Evaluate on test set
        print(f"🧪 Evaluating fold {fold+1} on test set...")
        test_metrics, test_true, test_probs = evaluate_comprehensive(model, test_loader, criterion, DEVICE)
        
        # Save ALL results for this fold
        print(f"💾 Saving results for fold {fold+1}...")
        
        # 1. Training history CSV
        save_fold_training_history(
            fold+1, train_losses, val_losses, train_accs, val_accs, OUT_DIR
        )
        
        # 2. Test predictions CSV
        save_fold_predictions(
            fold+1, test_true, test_probs, OUT_DIR
        )
        
        # 3. Confusion matrix CSV
        save_fold_confusion_matrix(
            fold+1, test_true, test_probs, OUT_DIR
        )
        
        # 4. SEPARATE training curves graphs
        plot_fold_loss_curve(fold+1, train_losses, val_losses, OUT_DIR)
        plot_fold_accuracy_curve(fold+1, train_accs, val_accs, OUT_DIR)
        
        # 5. ROC curve graph
        plot_fold_roc_curve(fold+1, test_true, test_probs, OUT_DIR)
        
        # 6. Metrics bar chart
        plot_fold_metrics(fold+1, test_metrics, OUT_DIR)
        
        # Store data for overall analysis - FIXED: Use the actual keys from test_metrics
        fold_result = test_metrics.copy()
        fold_result['fold'] = fold + 1
        fold_result['train_samples'] = len(train_df)
        fold_result['test_samples'] = len(test_df)
        all_fold_metrics.append(fold_result)
        
        # Store training history for convergence plots
        all_fold_histories.append({
            'train_loss': train_losses,
            'val_loss': val_losses,
            'train_acc': train_accs,
            'val_acc': val_accs
        })
        
        # Store data for overall ROC
        all_fold_data.append({
            'true_labels': test_true,
            'probs': test_probs
        })
        
        print(f"✅ Fold {fold+1} completed and saved!")
    
    # Save overall cross-validation results
    print(f"\n{'='*60}")
    print(f"📊 SAVING OVERALL CROSS-VALIDATION RESULTS")
    print(f"{'='*60}")
    
    # 1. Save combined metrics (ALL folds in one file)
    save_all_folds_metrics_combined(all_fold_metrics, OUT_DIR)
    
    # 2. Calculate confidence intervals
    confidence_intervals = calculate_confidence_intervals(all_fold_metrics)
    
    # 3. Save overall summary files
    summary_df, stats_df = save_overall_results(all_fold_metrics, confidence_intervals, OUT_DIR)
    
    # Generate overall graphs
    print(f"🖼️ Generating overall analysis graphs...")
    
    # 1. Mean ROC curve with std
    plot_mean_roc_curve(all_fold_data, OUT_DIR)
    
    # 2. Metrics distribution box plot
    plot_metrics_distribution(all_fold_metrics, OUT_DIR)
    
    # 3. SEPARATE training convergence graphs
    plot_training_convergence_separate(all_fold_histories, OUT_DIR)
    
    # 4. Training stability plot
    plot_training_stability(all_fold_histories, OUT_DIR)
    
    # Print final summary
    print(f"\n🎯 SCENARIO 1 COMPLETED!")
    print(f"\n📈 OVERALL PERFORMANCE (Mean ± Std across 5 folds):")
    for metric in ['accuracy', 'f1', 'precision', 'recall', 'specificity', 'auc']:
        mean_val = stats_df[stats_df['metric'] == f'test_{metric}']['mean'].values[0]
        std_val = stats_df[stats_df['metric'] == f'test_{metric}']['std'].values[0]
        print(f"   {metric.upper():12s}: {mean_val:.4f} ± {std_val:.4f}")
    
    print(f"\n📁 ALL RESULTS SAVED TO: {OUT_DIR}")
    
    # Print file summary
    print(f"\n📋 GENERATED FILES SUMMARY:")
    print(f"\n📊 For EACH Fold (1-5):")
    print(f"   CSV Files:")
    print(f"   - fold_X_training_history.csv")
    print(f"   - fold_X_predictions.csv") 
    print(f"   - fold_X_confusion_matrix.csv")
    print(f"   Graph Files:")
    print(f"   - fold_X_loss_curve.png")
    print(f"   - fold_X_accuracy_curve.png")
    print(f"   - fold_X_roc_curve.png")
    print(f"   - fold_X_metrics.png")
    
    print(f"\n📈 Overall Summary Files:")
    print(f"   CSV Files:")
    print(f"   - all_folds_metrics_combined.csv")
    print(f"   - scenario1_all_folds_summary.csv")
    print(f"   - scenario1_cross_validation_statistics.csv")
    print(f"   Graph Files:")
    print(f"   - scenario1_mean_roc_curve.png")
    print(f"   - scenario1_metrics_distribution.png") 
    print(f"   - scenario1_accuracy_convergence.png")
    print(f"   - scenario1_loss_convergence.png")
    print(f"   - scenario1_training_stability.png")
    
    print(f"\n🎯 Total: 18 CSV files + 25 graph files = 43 files")
    
    return all_fold_metrics, summary_df, stats_df

if __name__ == "__main__":
    all_metrics, summary, statistics = main()