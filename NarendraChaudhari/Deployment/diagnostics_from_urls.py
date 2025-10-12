"""
Diagnostics script
- Place direct-download URLs (one per line) in `urls.txt` in the project root.
- Run: python diagnostics_from_urls.py
- The script downloads images, loads `haar_hybridcnn.pth` from the project root,
  runs inference with different preprocessing options and temperatures, and
  writes `diagnostics_results.csv` with top-5 predictions per image/setting.

Note: URLs must be direct links (Dropbox ?dl=1, Google Drive uc?export=download&id=FILEID, raw GitHub links, etc.).
"""
import os
import sys
import requests
from io import BytesIO
from PIL import Image
import torch
import torch.nn as nn
import numpy as np
import csv

ROOT = os.path.dirname(__file__)
CKPT = os.path.join(ROOT, "haar_hybridcnn.pth")
URLS_FILE = os.path.join(ROOT, "urls.txt")
OUT_CSV = os.path.join(ROOT, "diagnostics_results.csv")

# --- model definition (same as in app.py) ---
class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)
    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += identity
        out = self.relu(out)
        return out

class HybridCNN(nn.Module):
    def __init__(self, num_classes):
        super(HybridCNN, self).__init__()
        self.initial = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.res1 = ResidualBlock(32)
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.res2 = ResidualBlock(64)
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1,1)),
            nn.Flatten(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, num_classes)
        )
    def forward(self, x):
        x = self.initial(x)
        x = self.res1(x)
        x = self.conv2(x)
        x = self.res2(x)
        x = self.conv3(x)
        x = self.classifier(x)
        return x


def load_checkpoint(path):
    device = torch.device('cpu')
    raw = torch.load(path, map_location=device)
    if isinstance(raw, dict) and 'state_dict' in raw:
        sd = raw['state_dict']
    elif isinstance(raw, dict):
        sd = raw
    else:
        sd = None
    # infer num_classes
    num_classes = 2
    if sd is not None:
        classifier_weight_keys = [k for k in sd.keys() if 'classifier' in k and k.endswith('.weight')]
        if classifier_weight_keys:
            key = sorted(classifier_weight_keys)[-1]
            num_classes = int(sd[key].shape[0])
    model = HybridCNN(num_classes=num_classes)
    if sd is not None:
        # handle module. prefix
        from collections import OrderedDict
        if any(k.startswith('module.') for k in sd.keys()):
            sd = OrderedDict((k.replace('module.',''), v) for k, v in sd.items())
        model.load_state_dict(sd, strict=False)
    model.eval()
    return model


def download_image(url):
    try:
        r = requests.get(url, timeout=30)
        r.raise_for_status()
        try:
            return Image.open(BytesIO(r.content)).convert('RGB')
        except Exception:
            # try tifffile for tif/tiff
            try:
                import tifffile
                arr = tifffile.imread(BytesIO(r.content))
                return Image.fromarray(arr.astype('uint8')).convert('RGB')
            except Exception as e:
                raise
    except Exception as e:
        print(f"Failed to download {url}: {e}")
        return None

from torchvision import transforms

def make_tf(option, size=(128,128)):
    if option == 'ImageNet':
        return transforms.Compose([transforms.Resize(size), transforms.ToTensor(), transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])
    if option == 'No':
        return transforms.Compose([transforms.Resize(size), transforms.ToTensor()])
    if option == '0.5':
        return transforms.Compose([transforms.Resize(size), transforms.ToTensor(), transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])])
    return transforms.Compose([transforms.Resize(size), transforms.ToTensor()])


def topk_from_probs(probs, labels, k=5):
    idxs = np.argsort(probs)[::-1][:k]
    return [(labels[i] if i < len(labels) else f'class_{i}', float(probs[i])) for i in idxs]


def main():
    if not os.path.exists(URLS_FILE):
        print(f"Create {URLS_FILE} with direct-download URLs (one per line).")
        return
    with open(URLS_FILE, 'r', encoding='utf-8') as f:
        urls = [l.strip() for l in f if l.strip()]
    if not urls:
        print('No URLs found in urls.txt')
        return
    if not os.path.exists(CKPT):
        print(f'Checkpoint not found at {CKPT}')
        return

    model = load_checkpoint(CKPT)

    labels_path = os.path.join(ROOT, 'labels.json')
    if os.path.exists(labels_path):
        import json
        with open(labels_path, 'r', encoding='utf-8') as f:
            labels = json.load(f)
    else:
        labels = None

    preproc_options = ['ImageNet', 'No', '0.5']
    temps = [1.0, 2.0, 5.0]

    rows = []
    for url in urls:
        print('\n===', url)
        img = download_image(url)
        if img is None:
            continue
        for opt in preproc_options:
            tf = make_tf(opt)
            inp = tf(img).unsqueeze(0)
            for T in temps:
                with torch.no_grad():
                    logits = model(inp)
                    logits = logits / float(T)
                    probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
                top5 = topk_from_probs(probs, labels or [], 5)
                print(f"preproc={opt}, T={T} -> top5: {top5}")
                rows.append([url, opt, T, ';'.join([f"{t[0]}:{t[1]:.6f}" for t in top5])])

    # write CSV
    try:
        with open(OUT_CSV, 'w', newline='', encoding='utf-8') as f:
            w = csv.writer(f)
            w.writerow(['url','preproc','temp','top5'])
            w.writerows(rows)
        print('\nWrote results to', OUT_CSV)
    except Exception as e:
        print('Failed to write CSV:', e)

if __name__ == '__main__':
    main()
