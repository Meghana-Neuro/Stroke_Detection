# ===========================================================
# unified_temporal_experiment.py
# Single script for BiGRU, BiLSTM, TemporalAttention, LightTransformer
# Includes all original features: splits saving, paper CSV, summary print
# ===========================================================
import os, sys, copy, time, random, threading, subprocess
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, f1_score, precision_score, recall_score,
                             confusion_matrix, roc_auc_score, roc_curve,
                             matthews_corrcoef, cohen_kappa_score)
import matplotlib.pyplot as plt
from pathlib import Path
from torchvision import models, transforms
import torchvision.transforms.functional as TF
from scipy import stats
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# ================== FIXED CONFIGURATION ==================
SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
os.environ["PYTHONHASHSEED"] = str(SEED)

# ---- USER-EDITABLE SETTINGS ----
MODEL_TYPE = "BiGRU"          # Options: "BiGRU", "BiLSTM", "TemporalAttention", "LightTransformer"
USE_SCHEDULER = True          # Toggle scheduler
USE_AUGMENTATION = True    # Keep False to match BiGRU baseline

# ---- PATHS ----
MANIFEST = "/home/m.byalalu/Face_preprocessing/RESULTS_5FPS/manifest_rf_crops_flat_fixed.csv"
SPLITS_DIR = "Unified_split"   # Folder containing fold_X_subjects.csv from BiGRU
BASE_OUT_DIR = "Unified_Results_BIGRU"

# ---- HYPERPARAMETERS (identical for all models) ----
BATCH_SIZE = 32
IMG_SIZE = 224
T = 16
LR = 1e-4
WEIGHT_DECAY = 1e-4
EPOCHS = 50
NUM_CLASSES = 2
HIDDEN_DIM = 128
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ---- AUTOMATED OUTPUT DIR ----
config_str = f"{MODEL_TYPE}_sched{USE_SCHEDULER}_aug{USE_AUGMENTATION}_seed{SEED}"
OUT_DIR = os.path.join(BASE_OUT_DIR, config_str)
Path(OUT_DIR).mkdir(parents=True, exist_ok=True)
UNIFIED_CSV = os.path.join(OUT_DIR, "computational_efficiency_overall.csv")

print(f"🚀 Running: {config_str}")
print(f"📁 Output: {OUT_DIR}")

# ================== MODEL COMPONENTS ==================
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
        return self.dropout(h)

# ---------- BiGRU Head ----------
class BiGRUHead(nn.Module):
    def __init__(self, in_dim=512, hidden=128, num_classes=2):
        super().__init__()
        self.gru = nn.GRU(in_dim, hidden, batch_first=True, bidirectional=True)
        self._init_gru_weights()
        self.drop1 = nn.Dropout(0.3)
        self.fc1 = nn.Linear(2 * hidden, hidden // 2)
        self.drop2 = nn.Dropout(0.3)
        self.fc2 = nn.Linear(hidden // 2, num_classes)
        self.relu = nn.ReLU()
    def _init_gru_weights(self):
        for name, param in self.gru.named_parameters():
            if 'weight_ih' in name:
                nn.init.orthogonal_(param.data)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param.data)
            elif 'bias' in name:
                param.data.fill_(0)
                n = param.size(0)
                param.data[n//3 : 2*n//3].fill_(1)
    def forward(self, seq):
        out, _ = self.gru(seq)
        last = out[:, -1]
        x = self.drop1(last)
        x = self.relu(self.fc1(x))
        x = self.drop2(x)
        return self.fc2(x)

# ---------- BiLSTM Head ----------
class BiLSTMHead(nn.Module):
    def __init__(self, in_dim=512, hidden=128, num_classes=2):
        super().__init__()
        self.lstm = nn.LSTM(in_dim, hidden, batch_first=True, bidirectional=True)
        self._init_lstm_weights()
        self.drop1 = nn.Dropout(0.3)
        self.fc1 = nn.Linear(2 * hidden, hidden // 2)
        self.drop2 = nn.Dropout(0.3)
        self.fc2 = nn.Linear(hidden // 2, num_classes)
        self.relu = nn.ReLU()
    def _init_lstm_weights(self):
        for name, param in self.lstm.named_parameters():
            if 'weight_ih' in name:
                nn.init.orthogonal_(param.data)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param.data)
            elif 'bias' in name:
                param.data.fill_(0)
                n = param.size(0)
                param.data[n//4 : n//2].fill_(1)
    def forward(self, seq):
        out, (h_n, c_n) = self.lstm(seq)
        last = torch.cat([h_n[0], h_n[1]], dim=1)
        x = self.drop1(last)
        x = self.relu(self.fc1(x))
        x = self.drop2(x)
        return self.fc2(x)

# ---------- Temporal Attention Head ----------
class TemporalAttentionHead(nn.Module):
    def __init__(self, in_dim=512, num_classes=2):
        super().__init__()
        self.attn = nn.Sequential(
            nn.Linear(in_dim, 128),
            nn.Tanh(),
            nn.Linear(128, 1)
        )
        self.classifier = nn.Linear(in_dim, num_classes)
    def forward(self, seq):
        weights = torch.softmax(self.attn(seq).squeeze(-1), dim=1)
        weighted = torch.sum(seq * weights.unsqueeze(-1), dim=1)
        return self.classifier(weighted)

# ---------- Light Transformer Head ----------
class LightTransformerHead(nn.Module):
    def __init__(self, in_dim=512, num_classes=2):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(d_model=in_dim, nhead=8, dim_feedforward=256, dropout=0.3, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.classifier = nn.Linear(in_dim, num_classes)
    def forward(self, seq):
        out = self.transformer(seq)
        pooled = out.mean(dim=1)
        return self.classifier(pooled)

# ================== FULL MODEL ==================
class FullModel(nn.Module):
    def __init__(self, model_type, hidden=128, num_classes=2):
        super().__init__()
        self.feats = ResNet18FeatureExtractor(dropout_rate=0.3)
        if model_type == "BiGRU":
            self.head = BiGRUHead(512, hidden, num_classes)
        elif model_type == "BiLSTM":
            self.head = BiLSTMHead(512, hidden, num_classes)
        elif model_type == "TemporalAttention":
            self.head = TemporalAttentionHead(512, num_classes)
        elif model_type == "LightTransformer":
            self.head = LightTransformerHead(512, num_classes)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
    def forward(self, x):
        B, T, C, H, W = x.shape
        f = self.feats(x.view(B * T, C, H, W)).view(B, T, 512)
        return self.head(f)

# ================== MODEL EFFICIENCY ==================
def compute_model_efficiency(model, device, T=16, img_size=224):
    model.eval()
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    param_size_mb = sum(p.numel() * p.element_size() for p in model.parameters()) / (1024**2)
    dummy = torch.zeros(1, T, 3, img_size, img_size).to(device)
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
    with torch.no_grad():
        _ = model(dummy)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        peak_fwd_mib = torch.cuda.max_memory_allocated() / (1024**2)
    else:
        peak_fwd_mib = 0.0
    latencies = []
    with torch.no_grad():
        for _ in range(50):
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            _ = model(dummy)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            latencies.append((time.perf_counter() - t0) * 1000)
    latency_mean_ms = float(np.mean(latencies[5:]))
    latency_std_ms = float(np.std(latencies[5:]))
    return {
        "total_params": total_params,
        "trainable_params": trainable_params,
        "model_size_mb": round(param_size_mb, 2),
        "peak_fwd_mib": round(peak_fwd_mib, 2),
        "inference_ms_mean": round(latency_mean_ms, 3),
        "inference_ms_std": round(latency_std_ms, 3),
        "clips_per_sec": round(1000 / latency_mean_ms, 2),
    }

# ================== DATASET ==================
def list_images_sorted(folder):
    exts = (".jpg", ".jpeg", ".png")
    try:
        return [str(Path(folder)/f) for f in sorted(os.listdir(folder)) if f.lower().endswith(exts)]
    except:
        return []

def sample_indices_tsn(num_frames, T, jitter=True):
    if num_frames <= 0: return []
    bounds = np.linspace(0, num_frames, T+1, dtype=int)
    idx = []
    for i in range(T):
        s, e = bounds[i], max(bounds[i+1]-1, bounds[i])
        if jitter and e>s:
            idx.append(np.random.randint(s, e+1))
        else:
            idx.append((s+e)//2)
    return idx

class VideoDataset(Dataset):
    def __init__(self, df, T=16, img_size=224, normalize=True, train_mode=True, use_aug=False):
        self.df = df.reset_index(drop=True)
        self.T = T
        self.img_size = img_size
        self.normalize = normalize
        self.train_mode = train_mode
        self.use_aug = use_aug and train_mode
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(3,1,1)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(3,1,1)
        self.augment = transforms.Compose([
            transforms.RandomResizedCrop(img_size, scale=(0.8, 1.0)),
            transforms.ColorJitter(0.2, 0.2, 0.2, 0.1),
            transforms.RandomRotation(10),
        ])
    def __len__(self): return len(self.df)
    def __getitem__(self, i):
        row = self.df.iloc[i]
        frames = list_images_sorted(row["clip_dir"])
        if len(frames) == 0:
            return torch.zeros(self.T,3,self.img_size,self.img_size), torch.tensor(row["label"], dtype=torch.long)
        idxs = sample_indices_tsn(len(frames), self.T, jitter=self.train_mode)
        if self.train_mode:
            seed = i + 123456
            random.seed(seed); torch.manual_seed(seed)
        imgs = []
        for j in idxs:
            p = frames[min(j, len(frames)-1)]
            try:
                from PIL import Image
                with Image.open(p) as im:
                    xpil = im.convert("RGB")
                    if self.use_aug:
                        xpil = self.augment(xpil)
                    else:
                        xpil = xpil.resize((self.img_size, self.img_size))
                    x = TF.to_tensor(xpil)
                    if self.normalize:
                        x = TF.normalize(x, mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
            except:
                x = torch.zeros(3, self.img_size, self.img_size)
            imgs.append(x)
        if self.train_mode:
            random.seed()
        return torch.stack(imgs, dim=0), torch.tensor(row["label"], dtype=torch.long)

# ================== TRAINING & EVALUATION ==================
def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss, correct, n = 0.0, 0, 0
    for X, y in loader:
        X, y = X.to(device), y.to(device)
        optimizer.zero_grad()
        out = model(X)
        loss = criterion(out, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item() * X.size(0)
        correct += (out.argmax(1) == y).sum().item()
        n += X.size(0)
    return total_loss / n, correct / n

def evaluate_comprehensive(model, loader, criterion, device):
    model.eval()
    total_loss, n = 0.0, 0
    all_true, all_pred, all_probs = [], [], []
    with torch.no_grad():
        for X, y in loader:
            X, y = X.to(device), y.to(device)
            out = model(X)
            loss = criterion(out, y)
            total_loss += loss.item() * X.size(0)
            n += X.size(0)
            probs = torch.softmax(out, dim=1)[:, 1]
            preds = out.argmax(1)
            all_true.extend(y.cpu().numpy())
            all_pred.extend(preds.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    acc = accuracy_score(all_true, all_pred)
    f1 = f1_score(all_true, all_pred, zero_division=0)
    prec = precision_score(all_true, all_pred, zero_division=0)
    rec = recall_score(all_true, all_pred, zero_division=0)
    cm = confusion_matrix(all_true, all_pred, labels=[0,1])
    if cm.shape == (2,2):
        tn, fp, fn, tp = cm.ravel()
        spec = tn / (tn+fp) if (tn+fp)>0 else 0
    else:
        tn=fp=fn=tp=0; spec=0
    mcc = matthews_corrcoef(all_true, all_pred)
    kappa = cohen_kappa_score(all_true, all_pred)
    bal_acc = (rec + spec)/2
    auc = roc_auc_score(all_true, all_probs) if len(np.unique(all_true))==2 else float('nan')
    return {
        'loss': total_loss/n,
        'accuracy': acc, 'f1': f1, 'precision': prec, 'recall': rec,
        'specificity': spec, 'auc': auc, 'mcc': mcc, 'kappa': kappa,
        'balanced_accuracy': bal_acc,
        'tp': int(tp), 'tn': int(tn), 'fp': int(fp), 'fn': int(fn)
    }, np.array(all_true), np.array(all_probs)

# ================== GPU MONITORING ==================
class GPUUtilizationSampler:
    def __init__(self, interval_sec=0.5):
        self.interval = interval_sec
        self._stop = threading.Event()
        self._thread = None
        self.samples = []
    def _tick(self):
        while not self._stop.is_set():
            try:
                out = subprocess.check_output(
                    ["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits", "-i", "0"],
                    stderr=subprocess.DEVNULL
                ).decode("utf-8").strip()
                self.samples.append(int(out.splitlines()[0].strip()))
            except:
                self.stop()
                break
            self._stop.wait(self.interval)
    def start(self):
        if self._thread is None or not self._thread.is_alive():
            self._stop.clear()
            self._thread = threading.Thread(target=self._tick, daemon=True)
            self._thread.start()
    def stop(self):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)

# ================== PLOTTING FUNCTIONS ==================
def plot_fold_loss_curve(fold_num, train_losses, val_losses, save_dir):
    plt.figure(figsize=(10,6))
    plt.plot(range(1, len(train_losses)+1), train_losses, 'b-', label='Train Loss')
    plt.plot(range(1, len(val_losses)+1), val_losses, 'r-', label='Val Loss')
    plt.title(f'Fold {fold_num} - Loss')
    plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.legend(); plt.grid(alpha=0.3)
    plt.savefig(os.path.join(save_dir, f"fold_{fold_num}_loss_curve.png"), dpi=300, bbox_inches='tight')
    plt.close()

def plot_fold_accuracy_curve(fold_num, train_accs, val_accs, save_dir):
    plt.figure(figsize=(10,6))
    plt.plot(range(1, len(train_accs)+1), train_accs, 'b-', label='Train Acc')
    plt.plot(range(1, len(val_accs)+1), val_accs, 'r-', label='Val Acc')
    plt.title(f'Fold {fold_num} - Accuracy'); plt.xlabel('Epoch'); plt.ylabel('Accuracy')
    plt.legend(); plt.grid(alpha=0.3); plt.ylim(0,1)
    plt.savefig(os.path.join(save_dir, f"fold_{fold_num}_accuracy_curve.png"), dpi=300, bbox_inches='tight')
    plt.close()

def plot_fold_roc_curve(fold_num, y_true, y_probs, save_dir):
    fpr, tpr, _ = roc_curve(y_true, y_probs)
    auc_score = roc_auc_score(y_true, y_probs)
    plt.figure(figsize=(8,8))
    plt.plot(fpr, tpr, 'darkorange', lw=2, label=f'AUC = {auc_score:.3f}')
    plt.plot([0,1],[0,1],'k--', alpha=0.5)
    plt.xlim([0,1]); plt.ylim([0,1.05]); plt.xlabel('FPR'); plt.ylabel('TPR')
    plt.title(f'Fold {fold_num} - ROC'); plt.legend(); plt.grid(alpha=0.3)
    plt.savefig(os.path.join(save_dir, f"fold_{fold_num}_roc_curve.png"), dpi=300, bbox_inches='tight')
    plt.close()

def plot_fold_metrics(fold_num, metrics, save_dir):
    names = ['accuracy','f1','precision','recall','specificity','auc']
    vals = [metrics[n] for n in names]
    plt.figure(figsize=(12,6))
    plt.bar(names, vals, color=['#1f77b4','#ff7f0e','#2ca02c','#d62728','#9467bd','#8c564b'])
    plt.ylim(0,1); plt.ylabel('Score'); plt.title(f'Fold {fold_num} - Test Metrics')
    plt.grid(axis='y', alpha=0.3)
    for i,v in enumerate(vals): plt.text(i, v+0.01, f'{v:.3f}', ha='center', fontweight='bold')
    plt.savefig(os.path.join(save_dir, f"fold_{fold_num}_metrics.png"), dpi=300, bbox_inches='tight')
    plt.close()

def plot_mean_roc_curve(all_fold_data, save_dir):
    plt.figure(figsize=(8,8))
    tprs, aucs = [], []
    mean_fpr = np.linspace(0,1,100)
    for data in all_fold_data:
        fpr, tpr, _ = roc_curve(data['true'], data['probs'])
        aucs.append(roc_auc_score(data['true'], data['probs']))
        tprs.append(np.interp(mean_fpr, fpr, tpr))
        tprs[-1][0]=0.0
    mean_tpr = np.mean(tprs, axis=0); mean_tpr[-1]=1.0
    std_tpr = np.std(tprs, axis=0)
    mean_auc = np.mean(aucs); std_auc = np.std(aucs)
    plt.plot(mean_fpr, mean_tpr, 'b', lw=2, label=f'Mean AUC = {mean_auc:.3f} ± {std_auc:.3f}')
    plt.fill_between(mean_fpr, mean_tpr-std_tpr, mean_tpr+std_tpr, color='grey', alpha=0.3)
    plt.plot([0,1],[0,1],'k--',alpha=0.5); plt.xlim([0,1]); plt.ylim([0,1.05])
    plt.xlabel('FPR'); plt.ylabel('TPR'); plt.title('Mean ROC Curve'); plt.legend(); plt.grid(alpha=0.3)
    plt.savefig(os.path.join(save_dir, "mean_roc_curve.png"), dpi=300, bbox_inches='tight')
    plt.close()

def plot_metrics_distribution(all_fold_metrics, save_dir):
    metrics_names = ['accuracy','f1','precision','recall','specificity','auc']
    data = {m: [fold[m] for fold in all_fold_metrics] for m in metrics_names}
    plt.figure(figsize=(12,6))
    plt.boxplot([data[m] for m in metrics_names], labels=metrics_names)
    plt.ylabel('Score'); plt.title('Metrics Distribution Across Folds'); plt.grid(axis='y', alpha=0.3)
    plt.savefig(os.path.join(save_dir, "metrics_distribution.png"), dpi=300, bbox_inches='tight')
    plt.close()

def plot_training_convergence(all_fold_histories, save_dir):
    max_epochs = max(len(h['train_acc']) for h in all_fold_histories)
    all_train_acc = np.full((len(all_fold_histories), max_epochs), np.nan)
    all_val_acc = np.full_like(all_train_acc, np.nan)
    all_train_loss = np.full_like(all_train_acc, np.nan)
    all_val_loss = np.full_like(all_train_acc, np.nan)
    for i, h in enumerate(all_fold_histories):
        e = len(h['train_acc'])
        all_train_acc[i,:e] = h['train_acc']
        all_val_acc[i,:e] = h['val_acc']
        all_train_loss[i,:e] = h['train_loss']
        all_val_loss[i,:e] = h['val_loss']
    epochs = range(1, max_epochs+1)
    # Accuracy
    plt.figure(figsize=(10,6))
    plt.plot(epochs, np.nanmean(all_train_acc, axis=0), 'b-', lw=3, label='Train')
    plt.plot(epochs, np.nanmean(all_val_acc, axis=0), 'r-', lw=3, label='Val')
    plt.fill_between(epochs, np.nanmean(all_train_acc,0)-np.nanstd(all_train_acc,0), np.nanmean(all_train_acc,0)+np.nanstd(all_train_acc,0), color='b', alpha=0.2)
    plt.fill_between(epochs, np.nanmean(all_val_acc,0)-np.nanstd(all_val_acc,0), np.nanmean(all_val_acc,0)+np.nanstd(all_val_acc,0), color='r', alpha=0.2)
    plt.xlabel('Epoch'); plt.ylabel('Accuracy'); plt.title('Accuracy Convergence'); plt.legend(); plt.grid(alpha=0.3)
    plt.savefig(os.path.join(save_dir, "accuracy_convergence.png"), dpi=300, bbox_inches='tight')
    plt.close()
    # Loss
    plt.figure(figsize=(10,6))
    plt.plot(epochs, np.nanmean(all_train_loss, axis=0), 'b-', lw=3, label='Train')
    plt.plot(epochs, np.nanmean(all_val_loss, axis=0), 'r-', lw=3, label='Val')
    plt.fill_between(epochs, np.nanmean(all_train_loss,0)-np.nanstd(all_train_loss,0), np.nanmean(all_train_loss,0)+np.nanstd(all_train_loss,0), color='b', alpha=0.2)
    plt.fill_between(epochs, np.nanmean(all_val_loss,0)-np.nanstd(all_val_loss,0), np.nanmean(all_val_loss,0)+np.nanstd(all_val_loss,0), color='r', alpha=0.2)
    plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.title('Loss Convergence'); plt.legend(); plt.grid(alpha=0.3)
    plt.savefig(os.path.join(save_dir, "loss_convergence.png"), dpi=300, bbox_inches='tight')
    plt.close()

def plot_training_stability(all_fold_histories, save_dir):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    for i, h in enumerate(all_fold_histories):
        epochs = range(1, len(h['train_acc'])+1)
        ax1.plot(epochs, h['train_acc'], 'b-', alpha=0.3, lw=1)
        ax1.plot(epochs, h['val_acc'], 'r-', alpha=0.3, lw=1)
        ax2.plot(epochs, h['train_loss'], 'b-', alpha=0.3, lw=1)
        ax2.plot(epochs, h['val_loss'], 'r-', alpha=0.3, lw=1)
    ax1.set_xlabel('Epoch'); ax1.set_ylabel('Accuracy'); ax1.set_title('Training Stability - Accuracy')
    ax2.set_xlabel('Epoch'); ax2.set_ylabel('Loss'); ax2.set_title('Training Stability - Loss')
    ax1.grid(alpha=0.3); ax2.grid(alpha=0.3)
    plt.savefig(os.path.join(save_dir, "training_stability.png"), dpi=300, bbox_inches='tight')
    plt.close()

# ================== CSV SAVING FUNCTIONS ==================
def save_fold_history(fold, train_loss, val_loss, train_acc, val_acc, out_dir):
    pd.DataFrame({'epoch':range(1,len(train_loss)+1), 'train_loss':train_loss, 'val_loss':val_loss,
                  'train_acc':train_acc, 'val_acc':val_acc}).to_csv(os.path.join(out_dir, f"fold_{fold}_history.csv"), index=False)
def save_fold_predictions(fold, y_true, y_prob, out_dir):
    pd.DataFrame({'true':y_true, 'prob':y_prob, 'pred':(y_prob>0.5).astype(int)}).to_csv(os.path.join(out_dir, f"fold_{fold}_preds.csv"), index=False)
def save_fold_cm(fold, y_true, y_prob, out_dir):
    cm = confusion_matrix(y_true, (y_prob>0.5).astype(int), labels=[0,1])
    pd.DataFrame(cm, index=['True 0','True 1'], columns=['Pred 0','Pred 1']).to_csv(os.path.join(out_dir, f"fold_{fold}_cm.csv"))
def calculate_confidence_intervals(metrics_list, conf=0.95):
    res = {}
    for metric in ['accuracy','f1','precision','recall','specificity','auc','mcc','kappa','balanced_accuracy']:
        vals = [m[metric] for m in metrics_list]
        mean, std = np.mean(vals), np.std(vals)
        se = std / np.sqrt(len(vals))
        ci = stats.t.interval(conf, len(vals)-1, loc=mean, scale=se)
        res[metric] = {'mean':mean, 'std':std, 'ci_lower':ci[0], 'ci_upper':ci[1], 'ci_width':ci[1]-ci[0]}
    return res
def save_all_metrics(metrics_list, conf_intervals, out_dir):
    # Combined per fold
    rows = []
    for m in metrics_list:
        rows.append({'fold':m['fold'], 'accuracy':m['accuracy'], 'f1':m['f1'], 'precision':m['precision'],
                     'recall':m['recall'], 'specificity':m['specificity'], 'auc':m['auc'], 'mcc':m['mcc'],
                     'kappa':m['kappa'], 'balanced_accuracy':m['balanced_accuracy'], 'loss':m['loss'],
                     'tp':m['tp'], 'tn':m['tn'], 'fp':m['fp'], 'fn':m['fn'],
                     'train_samples':m.get('train_samples',0), 'test_samples':m.get('test_samples',0)})
    pd.DataFrame(rows).to_csv(os.path.join(out_dir, "all_folds_metrics.csv"), index=False)
    # Statistics
    stats_rows = []
    for metric in conf_intervals:
        ci = conf_intervals[metric]
        vals = [m[metric] for m in metrics_list]
        stats_rows.append({'metric':'test_'+metric, 'mean':ci['mean'], 'std':ci['std'], 'min':np.min(vals), 'max':np.max(vals),
                           'ci_lower':ci['ci_lower'], 'ci_upper':ci['ci_upper'], 'ci_width':ci['ci_width']})
    pd.DataFrame(stats_rows).to_csv(os.path.join(out_dir, "cross_validation_statistics.csv"), index=False)

# ================== HELPER FUNCTIONS ==================
def _has_both_classes(df: pd.DataFrame) -> bool:
    return set(df['label'].unique()) == {0, 1}

# ================== MAIN ==================
def main():
    # Load manifest
    df = pd.read_csv(MANIFEST)
    
    # Subject to label mapping
    subject_label_map = df.groupby("subject_id")["label"].first().to_dict()
    subjects = np.array(list(subject_label_map.keys()))
    subject_labels = np.array([subject_label_map[s] for s in subjects])
    
    print(f"\n📁 Data Overview:")
    print(f"   Total samples: {len(df)}")
    print(f"   Total unique subjects: {len(subjects)}")
    print(f"   Classes: {df['label'].value_counts().to_dict()}")
    
    # ================== HANDLE SPLITS ==================
    # Check if split files already exist
    split_files_exist = all(
        os.path.exists(os.path.join(SPLITS_DIR, f"fold_{fold}_subjects.csv"))
        for fold in range(1, 6)
    )

    if split_files_exist:
        print(f"✅ Loading existing splits from {SPLITS_DIR}")
        folds_data = []
        for fold in range(1, 6):
            split_df = pd.read_csv(os.path.join(SPLITS_DIR, f"fold_{fold}_subjects.csv"))
            train_subs = [s.strip() for s in split_df["train_subjects"].iloc[0].split(",")]
            val_subs   = [s.strip() for s in split_df["val_subjects"].iloc[0].split(",")]
            test_subs  = [s.strip() for s in split_df["test_subjects"].iloc[0].split(",")]
            folds_data.append((train_subs, val_subs, test_subs))
    else:
        print(f"🆕 No existing splits found. Generating new 5-fold splits...")
        Path(SPLITS_DIR).mkdir(parents=True, exist_ok=True)
        
        from sklearn.model_selection import StratifiedGroupKFold
        kfold = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
        folds_data = []
        
        for fold, (train_val_idx, test_idx) in enumerate(
            kfold.split(X=subjects, y=subject_labels, groups=subjects)
        ):
            train_val_subjects = subjects[train_val_idx]
            test_subjects = subjects[test_idx]
            tv_labels = np.array([subject_label_map[s] for s in train_val_subjects])
            train_subjects, val_subjects = train_test_split(
                train_val_subjects,
                test_size=0.2,
                random_state=SEED,
                stratify=tv_labels
            )
            train_subs = train_subjects.tolist()
            val_subs = val_subjects.tolist()
            test_subs = test_subjects.tolist()
            folds_data.append((train_subs, val_subs, test_subs))
            
            # Save the split for future reuse
            pd.DataFrame({
                "train_subjects": [",".join(train_subs)],
                "val_subjects":   [",".join(val_subs)],
                "test_subjects":  [",".join(test_subs)],
            }).to_csv(os.path.join(SPLITS_DIR, f"fold_{fold+1}_subjects.csv"), index=False)
            print(f"   Saved fold {fold+1} split to {SPLITS_DIR}")

    # ================== MODEL EFFICIENCY ==================
    # ... rest of main() indented inside ...
    
    # ================== MODEL EFFICIENCY ==================
    print(f"\n📐 Computing model efficiency metrics...")
    _eff_model = FullModel(MODEL_TYPE, HIDDEN_DIM, NUM_CLASSES).to(DEVICE)
    eff = compute_model_efficiency(_eff_model, DEVICE, T, IMG_SIZE)
    del _eff_model
    pd.DataFrame([eff]).to_csv(os.path.join(OUT_DIR, "model_efficiency.csv"), index=False)
    print(f"   Total params: {eff['total_params']:,} | Inference: {eff['inference_ms_mean']} ms")
    
    # ================== RUN FOLDS ==================
    all_fold_metrics = []
    all_fold_histories = []
    all_fold_data = []
    overall_epoch_times = []
    overall_wall_clock = 0.0
    overall_train_time = 0.0
    overall_clips_seen = 0
    overall_peak_gpu_mib = 0.0
    overall_gpu_utils = []
    
    for fold, (train_subs, val_subs, test_subs) in enumerate(folds_data):
        print(f"\n{'='*50}\nFOLD {fold+1}/5\n{'='*50}")
        train_df = df[df['subject_id'].isin(train_subs)].reset_index(drop=True)
        val_df   = df[df['subject_id'].isin(val_subs)].reset_index(drop=True)
        test_df  = df[df['subject_id'].isin(test_subs)].reset_index(drop=True)
        
        # Save subject splits for this fold (inside the run's output directory)
        pd.DataFrame({
            "train_subjects": [",".join(map(str, train_subs))],
            "val_subjects":   [",".join(map(str, val_subs))],
            "test_subjects":  [",".join(map(str, test_subs))],
        }).to_csv(os.path.join(OUT_DIR, f"fold_{fold+1}_subjects.csv"), index=False)
        pd.DataFrame({"subject_id": test_subs}).to_csv(
            os.path.join(OUT_DIR, f"fold_{fold+1}_test_subject_ids.csv"), index=False)
        
        if not _has_both_classes(val_df) or not _has_both_classes(test_df):
            print("⚠️  Single-class split detected in val/test.")
        
        print(f"📈 Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")
        
        train_loader = DataLoader(VideoDataset(train_df, T, IMG_SIZE, train_mode=True, use_aug=USE_AUGMENTATION),
                                  batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
        val_loader   = DataLoader(VideoDataset(val_df, T, IMG_SIZE, train_mode=False),
                                  batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
        test_loader  = DataLoader(VideoDataset(test_df, T, IMG_SIZE, train_mode=False),
                                  batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
        
        model = FullModel(MODEL_TYPE, HIDDEN_DIM, NUM_CLASSES).to(DEVICE)
        optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6) if USE_SCHEDULER else None
        
        class_counts = np.bincount(train_df['label'].values)
        class_weights = 1.0 / (class_counts + 1e-8)
        class_weights = class_weights / class_weights.sum() * len(class_counts)
        class_weights = torch.tensor(class_weights, dtype=torch.float).to(DEVICE)
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        
        sampler = GPUUtilizationSampler()
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            sampler.start()
        
        train_losses, val_losses = [], []
        train_accs, val_accs = [], []
        best_val_f1 = -1.0
        best_state = copy.deepcopy(model.state_dict())
        
        fold_start = time.time()
        for epoch in range(EPOCHS):
            t0 = time.time()
            train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, DEVICE)
            epoch_sec = time.time() - t0
            overall_epoch_times.append(epoch_sec)
            overall_train_time += epoch_sec
            overall_clips_seen += len(train_df)
            if torch.cuda.is_available():
                peak = torch.cuda.max_memory_allocated() / (1024**2)
                overall_peak_gpu_mib = max(overall_peak_gpu_mib, peak)
            
            val_metrics, _, _ = evaluate_comprehensive(model, val_loader, criterion, DEVICE)
            val_loss, val_acc, val_f1 = val_metrics['loss'], val_metrics['accuracy'], val_metrics['f1']
            if scheduler is not None:
                scheduler.step()
            
            train_losses.append(train_loss); train_accs.append(train_acc)
            val_losses.append(val_loss); val_accs.append(val_acc)
            if val_f1 >= best_val_f1:
                best_val_f1 = val_f1
                best_state = copy.deepcopy(model.state_dict())
            if (epoch+1)%10 == 0:
                print(f"Epoch {epoch+1:3d} | Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}")
        
        sampler.stop()
        overall_gpu_utils.extend(sampler.samples)
        overall_wall_clock += (time.time() - fold_start)
        
        model.load_state_dict(best_state)
        test_metrics, test_true, test_probs = evaluate_comprehensive(model, test_loader, criterion, DEVICE)
        test_metrics['fold'] = fold+1
        test_metrics['train_samples'] = len(train_df)
        test_metrics['test_samples'] = len(test_df)
        all_fold_metrics.append(test_metrics)
        all_fold_histories.append({'train_loss':train_losses, 'val_loss':val_losses, 'train_acc':train_accs, 'val_acc':val_accs})
        all_fold_data.append({'true':test_true, 'probs':test_probs})
        
        # Save per-fold results
        save_fold_history(fold+1, train_losses, val_losses, train_accs, val_accs, OUT_DIR)
        save_fold_predictions(fold+1, test_true, test_probs, OUT_DIR)
        save_fold_cm(fold+1, test_true, test_probs, OUT_DIR)
        plot_fold_loss_curve(fold+1, train_losses, val_losses, OUT_DIR)
        plot_fold_accuracy_curve(fold+1, train_accs, val_accs, OUT_DIR)
        plot_fold_roc_curve(fold+1, test_true, test_probs, OUT_DIR)
        plot_fold_metrics(fold+1, test_metrics, OUT_DIR)
        print(f"✅ Fold {fold+1} complete. Test Acc: {test_metrics['accuracy']:.4f}, F1: {test_metrics['f1']:.4f}")
    
    # ================== SAVE OVERALL RESULTS ==================
    conf_intervals = calculate_confidence_intervals(all_fold_metrics)
    save_all_metrics(all_fold_metrics, conf_intervals, OUT_DIR)
    plot_mean_roc_curve(all_fold_data, OUT_DIR)
    plot_metrics_distribution(all_fold_metrics, OUT_DIR)
    plot_training_convergence(all_fold_histories, OUT_DIR)
    plot_training_stability(all_fold_histories, OUT_DIR)
    
    overall_row = {
        "setting": "centralized", "clients": 1, "rounds": 0, "local_epochs": EPOCHS, "folds": 5,
        "total_params": eff["total_params"], "model_size_mb": eff["model_size_mb"],
        "peak_fwd_mib": eff["peak_fwd_mib"], "inference_ms_mean": eff["inference_ms_mean"],
        "inference_ms_std": eff["inference_ms_std"],
        "wall_clock_h": overall_wall_clock/3600, "total_train_time_h": overall_train_time/3600,
        "wall_clock_per_fold_h": (overall_wall_clock/5)/3600, "train_time_per_fold_h": (overall_train_time/5)/3600,
        "avg_epoch_time_s": np.mean(overall_epoch_times), "std_epoch_time_s": np.std(overall_epoch_times),
        "peak_gpu_mib": overall_peak_gpu_mib,
        "throughput_clips_per_s": overall_clips_seen/overall_train_time if overall_train_time>0 else None,
        "total_train_clips": overall_clips_seen,
        "gpu_utilization_mean_pct": np.mean(overall_gpu_utils) if overall_gpu_utils else None,
        "gpu_utilization_std_pct": np.std(overall_gpu_utils) if overall_gpu_utils else None,
    }
    pd.DataFrame([overall_row]).to_csv(UNIFIED_CSV, mode='a', header=not os.path.exists(UNIFIED_CSV), index=False)
    
    paper_row = pd.DataFrame([{
        "Metric": "Centralized",
        "Wall-clock Time (h)": overall_row["wall_clock_h"],
        "Avg. Epoch Time (s)": overall_row["avg_epoch_time_s"],
        "Avg. Epoch Time Std (s)": overall_row["std_epoch_time_s"],
        "Peak GPU Memory (MiB)": overall_row["peak_gpu_mib"],
        "GPU Utilization Mean (%)": overall_row["gpu_utilization_mean_pct"],
        "GPU Utilization Std (%)": overall_row["gpu_utilization_std_pct"],
        "Throughput (clips/s)": overall_row["throughput_clips_per_s"],
    }])
    paper_csv = os.path.join(OUT_DIR, "scenario1_centralized_paper_metrics.csv")
    paper_row.to_csv(paper_csv, mode="a", index=False, header=not os.path.exists(paper_csv))
    
    print(f"\n🎯 ALL DONE. Results saved to {OUT_DIR}")
    
    # ADDED: Cosmetic file summary print
    print(f"\n📋 GENERATED FILES SUMMARY:")
    print(f"\n📊 For EACH Fold (1-5):")
    print(f"   CSV Files:")
    print(f"   - fold_X_history.csv")
    print(f"   - fold_X_preds.csv") 
    print(f"   - fold_X_cm.csv")
    print(f"   - fold_X_subjects.csv")
    print(f"   - fold_X_test_subject_ids.csv")
    print(f"   Graph Files:")
    print(f"   - fold_X_loss_curve.png")
    print(f"   - fold_X_accuracy_curve.png")
    print(f"   - fold_X_roc_curve.png")
    print(f"   - fold_X_metrics.png")
    print(f"\n📈 Overall Summary Files:")
    print(f"   CSV Files:")
    print(f"   - all_folds_metrics.csv")
    print(f"   - cross_validation_statistics.csv")
    print(f"   - model_efficiency.csv")
    print(f"   - computational_efficiency_overall.csv")
    print(f"   - scenario1_centralized_paper_metrics.csv")
    print(f"   Graph Files:")
    print(f"   - mean_roc_curve.png")
    print(f"   - metrics_distribution.png") 
    print(f"   - accuracy_convergence.png")
    print(f"   - loss_convergence.png")
    print(f"   - training_stability.png")
    print(f"\n🎯 All experiments completed. Results saved to {OUT_DIR}")

if __name__ == "__main__":
    main()