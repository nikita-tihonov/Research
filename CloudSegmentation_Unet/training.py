# ------------------------------------------------------------------------
# Download dataset

# import kagglehub

# path = kagglehub.dataset_download("sorour/38cloud-cloud-segmentation-in-satellite-images")

# print("Path to dataset files:", path)

# -------------------------------------------------------------------------
# Defining the functions

import os
import cv2
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, sampler
from torchvision import transforms
import numpy as np
import matplotlib.pyplot as plt
import segmentation_models_pytorch as smp
from tqdm import tqdm
from pathlib import Path
from PIL import Image
from skimage.segmentation import find_boundaries

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)


class CloudDataset(Dataset):
    def __init__(self, r_dir, g_dir, b_dir, nir_dir, gt_dir, pytorch=True):
        super().__init__()

        self.files = [self.combine_files(f, g_dir, b_dir, nir_dir, gt_dir) for f in r_dir.iterdir() if not f.is_dir()]

        self.pytorch = pytorch

    def combine_files(self, r_file: Path, g_dir, b_dir,nir_dir, gt_dir):

        files = {'red': r_file,
                 'green':g_dir/r_file.name.replace('red', 'green'),
                 'blue': b_dir/r_file.name.replace('red', 'blue'),
                 'nir': nir_dir/r_file.name.replace('red', 'nir'),
                 'gt': gt_dir/r_file.name.replace('red', 'gt')}

        return files

    def __len__(self):

        return len(self.files)

    def open_as_array(self, idx, invert=False):

        raw_rgb = np.stack([np.array(Image.open(self.files[idx]['red'])),
                            np.array(Image.open(self.files[idx]['green'])),
                            np.array(Image.open(self.files[idx]['blue'])),
                            np.array(Image.open(self.files[idx]['nir'])),
                           ], axis=2)

        if invert:
            raw_rgb = raw_rgb.transpose((2,0,1))

        # normalize
        return (raw_rgb / np.iinfo(raw_rgb.dtype).max)

    def open_mask(self, idx, add_dims=False):

        raw_mask = np.array(Image.open(self.files[idx]['gt']))
        raw_mask = np.where(raw_mask==255, 1, 0)

        return np.expand_dims(raw_mask, 0) if add_dims else raw_mask

    def __getitem__(self, idx):

        x = torch.tensor(self.open_as_array(idx, invert=self.pytorch), dtype=torch.float32)
        y = torch.tensor(self.open_mask(idx, add_dims=False), dtype=torch.torch.int64)

        return x, y

    def open_as_pil(self, idx):

        arr = 256*self.open_as_array(idx)

        return Image.fromarray(arr.astype(np.uint8), 'RGB')

    def __repr__(self):

        s = 'Dataset class with {} files'.format(self.__len__())

        return s


def pixel_accuracy(pred, target):
    correct = (pred == target).sum().item()
    total = target.numel()
    return correct / total


def mean_iou(pred, target, num_classes=2, ignore_index=None):
    ious = []
    pred = pred.view(-1)
    target = target.view(-1)
    
    if ignore_index is not None:
        valid = target != ignore_index
        pred, target = pred[valid], target[valid]

    for cls in range(num_classes):
        pred_inds = (pred == cls)
        target_inds = (target == cls)
        intersection = (pred_inds & target_inds).sum().item()
        union = (pred_inds | target_inds).sum().item()
        if union == 0:
            ious.append(float('nan'))  # игнорируем класс без примеров
        else:
            ious.append(intersection / union)
    return np.nanmean(ious)


def dice_coefficient(pred, target, num_classes=2, ignore_index=None):
    dices = []
    pred = pred.view(-1)
    target = target.view(-1)
    
    if ignore_index is not None:
        valid = target != ignore_index
        pred, target = pred[valid], target[valid]

    for cls in range(num_classes):
        pred_inds = (pred == cls)
        target_inds = (target == cls)
        intersection = (pred_inds & target_inds).sum().item()
        denom = pred_inds.sum().item() + target_inds.sum().item()
        if denom == 0:
            dices.append(float('nan'))
        else:
            dices.append(2.0 * intersection / denom)
    return np.nanmean(dices)


def frequency_weighted_iou(pred, target, num_classes=2, ignore_index=None):
    pred = pred.view(-1)
    target = target.view(-1)
    
    if ignore_index is not None:
        valid = target != ignore_index
        pred, target = pred[valid], target[valid]

    freqs = []
    ious = []
    total_pixels = target.numel()

    for cls in range(num_classes):
        target_inds = (target == cls)
        freq = target_inds.sum().item() / total_pixels
        if freq == 0:
            continue
        pred_inds = (pred == cls)
        intersection = (pred_inds & target_inds).sum().item()
        union = (pred_inds | target_inds).sum().item()
        iou = intersection / union if union > 0 else 0
        freqs.append(freq)
        ious.append(iou)
    
    if not freqs:
        return 0.0
    return np.average(ious, weights=freqs)


def entropy_score(logits):
    if logits.dim() == 4:
        logits = logits.squeeze(0)  # (C, H, W)
    probs = torch.softmax(logits, dim=0)
    log_probs = torch.log(probs + 1e-8)
    entropy = -(probs * log_probs).sum(dim=0).mean().item()
    return entropy


def boundary_f1(pred_mask, gt_mask, dilation=0):
    pred_boundary = find_boundaries(pred_mask, mode='thick')
    gt_boundary = find_boundaries(gt_mask, mode='thick')
    
    if dilation > 0:
        from skimage.morphology import binary_dilation, disk
        selem = disk(dilation)
        pred_boundary = binary_dilation(pred_boundary, selem)
        gt_boundary = binary_dilation(gt_boundary, selem)
    
    tp = np.sum(pred_boundary & gt_boundary)
    fp = np.sum(pred_boundary & ~gt_boundary)
    fn = np.sum(~pred_boundary & gt_boundary)
    
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    bf1 = 2 * precision * recall / (precision + recall + 1e-8)
    return bf1


def contour_accuracy(pred_mask, gt_mask, tolerance=2):
    pred_boundary = find_boundaries(pred_mask, mode='thick')
    gt_boundary = find_boundaries(gt_mask, mode='thick')
    
    from skimage.morphology import binary_dilation, disk
    dilated_gt = binary_dilation(gt_boundary, disk(tolerance))
    
    correct = np.sum(pred_boundary & dilated_gt)
    total_pred = np.sum(pred_boundary)
    return correct / (total_pred + 1e-8)


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    total_acc = 0.0
    num_batches = 0
    for images, masks in tqdm(loader, desc="Training"):
        images, masks = images.to(device), masks.to(device)
        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, masks)

        loss.backward()
        optimizer.step()
        
        # preds = torch.sigmoid(outputs) > 0.5
        preds = torch.argmax(outputs, dim=1)
        acc = pixel_accuracy(preds, masks)

        total_loss += loss.item()
        total_acc += acc
        num_batches += 1

    avg_loss = total_loss / num_batches
    avg_acc = total_acc / num_batches
    return avg_loss, avg_acc


def validate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    total_acc = 0.0
    num_batches = 0

    metrics = {
        'miou': [],
        'dice': [],
        'fw_iou': [],
        'entropy': [],
        'bf1': [],
        'contour_acc': []
    }

    with torch.no_grad():
        for images, masks in tqdm(loader, desc="Validating"):
            images, masks = images.to(device), masks.to(device)
            outputs = model(images)
            loss = criterion(outputs, masks)

            # preds = torch.sigmoid(outputs) > 0.5
            preds = torch.argmax(outputs, dim=1)
            acc = pixel_accuracy(preds, masks)

            total_loss += loss.item()
            total_acc += acc
            num_batches += 1

            for i in range(images.size(0)):

                pred = preds[i].cpu()
                mask = masks[i].cpu()
                logits = outputs[i].cpu()

                metrics['miou'].append(mean_iou(pred, mask, num_classes=2))
                metrics['dice'].append(dice_coefficient(pred, mask, num_classes=2))
                metrics['fw_iou'].append(frequency_weighted_iou(pred, mask, num_classes=2))
                metrics['entropy'].append(entropy_score(logits))

                pred_np = pred.numpy().astype(np.int32)
                target_np = mask.numpy().astype(np.int32)
                metrics['bf1'].append(boundary_f1(pred_np, target_np, dilation=2))
                metrics['contour_acc'].append(contour_accuracy(pred_np, target_np, tolerance=2))

    avg_loss = total_loss / num_batches
    avg_acc = total_acc / num_batches
    
    return avg_loss, avg_acc, {k: np.nanmean(v) for k, v in metrics.items()}

def visualize_training(history, save_path):

    epochs = range(1, len(history['train_loss']) + 1)

    plt.figure(figsize=(10, 20))

    plt.subplot(4, 2, 1)
    plt.plot(epochs, history['train_loss'], label='Train Loss')
    plt.plot(epochs, history['val_loss'], label='Val Loss')
    plt.title('Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    plt.subplot(4, 2, 2)
    plt.plot(epochs, history['train_acc'], label='Train Accuracy')
    plt.plot(epochs, history['val_acc'], label='Val Accuracy')
    plt.title('Pixel Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)

    plt.subplot(4, 2, 3)
    # plt.plot(epochs, history['train_acc'], label='Train Accuracy')
    plt.plot(epochs, history['miou'], label='Val Accuracy')
    plt.title('mIoU')
    plt.xlabel('Epochs')
    plt.ylabel('mIoU')
    plt.legend()
    plt.grid(True)

    plt.subplot(4, 2, 4)
    # plt.plot(epochs, history['train_acc'], label='Train Accuracy')
    plt.plot(epochs, history['dice'], label='Val Accuracy')
    plt.title('Dice Cofficient')
    plt.xlabel('Epochs')
    plt.ylabel('Dice Cofficient')
    plt.legend()
    plt.grid(True)

    plt.subplot(4, 2, 5)
    # plt.plot(epochs, history['train_acc'], label='Train Accuracy')
    plt.plot(epochs, history['fw_iou'], label='Val Accuracy')
    plt.title('fwIoU')
    plt.xlabel('Epochs')
    plt.ylabel('fwIoU')
    plt.legend()
    plt.grid(True)

    plt.subplot(4, 2, 6)
    # plt.plot(epochs, history['train_acc'], label='Train Accuracy')
    plt.plot(epochs, history['entropy'], label='Val Accuracy')
    plt.title('Entropy Score')
    plt.xlabel('Epochs')
    plt.ylabel('Score')
    plt.legend()
    plt.grid(True)

    plt.subplot(4, 2, 7)
    # plt.plot(epochs, history['train_acc'], label='Train Accuracy')
    plt.plot(epochs, history['bf1'], label='Val Accuracy')
    plt.title('Boundary F1 Score')
    plt.xlabel('Epochs')
    plt.ylabel('Score')
    plt.legend()
    plt.grid(True)

    plt.subplot(4, 2, 8)
    # plt.plot(epochs, history['train_acc'], label='Train Accuracy')
    plt.plot(epochs, history['contour_acc'], label='Val Accuracy')
    plt.title('Contour Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(save_path, dpi=100, bbox_inches='tight')
    plt.show()

# ----------------------------------------------------------------------------------------
# Training


base_path = Path('/root/.cache/kagglehub/datasets/sorour/38cloud-cloud-segmentation-in-satellite-images/versions/4/38-Cloud_training')
data = CloudDataset(base_path/'train_red',
                    base_path/'train_green',
                    base_path/'train_blue',
                    base_path/'train_nir',
                    base_path/'train_gt'
                    )
len(f"Количество снимков {data}")

train_ds, valid_ds = torch.utils.data.random_split(data, (6000, 2400))

train_loader = DataLoader(train_ds, batch_size=12, shuffle=True)
val_loader = DataLoader(valid_ds, batch_size=12, shuffle=True)

model = smp.Unet(
    encoder_name="resnet34",
    encoder_weights="imagenet",
    in_channels=4,
    classes=2
).to(device)

criterion = nn.CrossEntropyLoss()
# criterion = smp.losses.DiceLoss(mode='multiclass')
# criterion = nn.BCEWithLogitsLoss()

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

num_epochs = 25
best_val_loss = float('inf')

history = {
    'train_loss': [],
    'train_acc': [],
    'val_loss': [],
    'val_acc': [],
    'miou': [],
    'dice': [],
    'fw_iou': [],
    'entropy': [],
    'bf1': [],
    'contour_acc': []
}

for epoch in range(num_epochs):
    print(f"\nEpoch {epoch+1}/{num_epochs}")

    train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
    val_loss, val_acc, metrics = validate(model, val_loader, criterion, device)

    scheduler.step()

    print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
    print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

    history['train_loss'].append(train_loss)
    history['train_acc'].append(train_acc)
    history['val_loss'].append(val_loss)
    history['val_acc'].append(val_acc)
    history['miou'].append(metrics['miou'])
    history['dice'].append(metrics['dice'])
    history['fw_iou'].append(metrics['fw_iou'])
    history['entropy'].append(metrics['entropy'])
    history['bf1'].append(metrics['bf1'])
    history['contour_acc'].append(metrics['contour_acc'])

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), "weights/unet_cloud_25epochs.pth")
        print("Best model saved!")

visualize_training(history, "trainings/metrics_25epochs.png")