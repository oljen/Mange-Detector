"""
fox_mange_classifier.py
-----------------------
PHASE 1 STARTER: a healthy-vs-mange fox image classifier using transfer learning.

What it does:
  1. Loads images from two folders: data/healthy/ and data/mange/
  2. Splits them into train / validation / test sets
  3. Fine-tunes a pretrained ResNet18 to tell the two apart
  4. Handles class imbalance and reports RECALL on the mange class
     (the metric that matters most: missing a sick fox is the costly error)
  5. Saves the trained model and a picture of the mistakes it made

IMPORTANT: This is a feasibility probe, NOT a diagnostic tool. A model trained on
a few hundred images flags possibilities; a human (your rehab partner) decides.

------------------------------------------------------------------------------
SETUP
  pip install torch torchvision scikit-learn matplotlib pillow

FOLDER LAYOUT (put your images here before running):
  data/
    healthy/   <- healthy-fox images
    mange/     <- mange-affected fox images

RUN
  python fox_mange_classifier.py
------------------------------------------------------------------------------
"""

import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, models, transforms
from torchvision.models import ResNet18_Weights
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt

# ----------------------------- CONFIG ----------------------------------------
DATA_DIR    = "data"          # folder containing healthy/ and mange/
IMG_SIZE    = 224             # ResNet's expected input size
BATCH_SIZE  = 16
EPOCHS      = 10
LR          = 1e-3
VAL_FRAC    = 0.15            # fraction of data held out for validation
TEST_FRAC   = 0.15            # fraction held out for final test
SEED        = 42
MODEL_OUT   = "fox_mange_resnet18.pt"
MISTAKES_OUT = "test_mistakes.png"

# ImageNet normalisation stats (the pretrained model expects these)
MEAN = [0.485, 0.456, 0.406]
STD  = [0.229, 0.224, 0.225]


def set_seed(seed):
    """Make the run reproducible so results are comparable between experiments."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_transforms():
    """Augmentation for training (squeezes more value from a small dataset),
    plain resize/normalise for validation and test."""
    train_tf = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])
    eval_tf = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])
    return train_tf, eval_tf


def make_splits(data_dir, train_tf, eval_tf):
    """Build train/val/test sets.

    NOTE ON LEAKAGE: this does a RANDOM split. For a real dataset you should split
    so that images of the SAME individual fox never appear in both train and test,
    or your accuracy will look great and be a lie. Once your images are tagged by
    individual/location, group the indices by that tag and split on the groups.
    """
    # Two views of the same folder so train gets augmentation, eval does not.
    full_train = datasets.ImageFolder(data_dir, transform=train_tf)
    full_eval  = datasets.ImageFolder(data_dir, transform=eval_tf)

    n = len(full_train)
    indices = list(range(n))
    rng = random.Random(SEED)
    rng.shuffle(indices)

    n_test = int(n * TEST_FRAC)
    n_val  = int(n * VAL_FRAC)
    test_idx  = indices[:n_test]
    val_idx   = indices[n_test:n_test + n_val]
    train_idx = indices[n_test + n_val:]

    train_ds = Subset(full_train, train_idx)
    val_ds   = Subset(full_eval,  val_idx)
    test_ds  = Subset(full_eval,  test_idx)

    return train_ds, val_ds, test_ds, full_train, train_idx


def class_weights(full_dataset, train_idx, device):
    """Weight the loss by inverse class frequency so the model doesn't ignore the
    rarer class. Mange images are usually the scarce ones, and they're the ones
    we most need to catch."""
    targets = [full_dataset.targets[i] for i in train_idx]
    counts = np.bincount(targets, minlength=len(full_dataset.classes))
    counts = np.where(counts == 0, 1, counts)        # avoid divide-by-zero
    weights = counts.sum() / (len(counts) * counts)  # higher weight = rarer class
    return torch.tensor(weights, dtype=torch.float32, device=device)


def build_model(num_classes, device):
    """Pretrained ResNet18 with the backbone frozen; we train only the new head.
    Freezing suits a small dataset. Once you have thousands of images, try
    unfreezing the later layers for a bit more accuracy."""
    model = models.resnet18(weights=ResNet18_Weights.DEFAULT)
    for param in model.parameters():
        param.requires_grad = False
    model.fc = nn.Linear(model.fc.in_features, num_classes)  # new, trainable head
    return model.to(device)


def run_epoch(model, loader, criterion, optimizer, device, train):
    model.train() if train else model.eval()
    total_loss, correct, total = 0.0, 0, 0
    torch.set_grad_enabled(train)
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        if train:
            optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        if train:
            loss.backward()
            optimizer.step()
        total_loss += loss.item() * images.size(0)
        correct += (outputs.argmax(1) == labels).sum().item()
        total += images.size(0)
    return total_loss / total, correct / total


def evaluate(model, loader, device, class_names):
    """Final report on held-out data: per-class precision/recall/F1 + confusion matrix."""
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for images, labels in loader:
            preds = model(images.to(device)).argmax(1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())
    print("\n=== TEST RESULTS ===")
    print(classification_report(all_labels, all_preds, target_names=class_names, zero_division=0))
    print("Confusion matrix (rows = true, cols = predicted):")
    print("            " + "  ".join(f"{c:>8}" for c in class_names))
    cm = confusion_matrix(all_labels, all_preds, labels=range(len(class_names)))
    for name, row in zip(class_names, cm):
        print(f"{name:>10}  " + "  ".join(f"{v:>8}" for v in row))
    # The number to watch: recall on the mange class.
    if "mange" in class_names:
        m = class_names.index("mange")
        recall = cm[m, m] / cm[m].sum() if cm[m].sum() else 0.0
        print(f"\n>> Recall on 'mange' (share of sick foxes caught): {recall:.0%}")
        print("   Aim to push this high even at the cost of a few false alarms.")


def save_mistakes(model, test_ds, device, class_names, max_images=12):
    """Save a grid of misclassified test images, because looking at the mistakes
    teaches you more than the accuracy number does."""
    model.eval()
    wrong = []
    with torch.no_grad():
        for i in range(len(test_ds)):
            image, label = test_ds[i]
            pred = model(image.unsqueeze(0).to(device)).argmax(1).item()
            if pred != label:
                wrong.append((image, label, pred))
            if len(wrong) >= max_images:
                break
    if not wrong:
        print("\nNo misclassified test images to show — nice.")
        return
    cols = 4
    rows = (len(wrong) + cols - 1) // cols
    mean, std = np.array(MEAN), np.array(STD)
    plt.figure(figsize=(cols * 3, rows * 3))
    for idx, (image, label, pred) in enumerate(wrong):
        img = image.permute(1, 2, 0).cpu().numpy() * std + mean  # un-normalise
        img = np.clip(img, 0, 1)
        ax = plt.subplot(rows, cols, idx + 1)
        ax.imshow(img)
        ax.set_title(f"true: {class_names[label]}\npred: {class_names[pred]}", fontsize=9)
        ax.axis("off")
    plt.tight_layout()
    plt.savefig(MISTAKES_OUT, dpi=120)
    print(f"\nSaved {len(wrong)} misclassified examples to {MISTAKES_OUT} — go look at them.")


def main():
    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    if not Path(DATA_DIR).exists():
        raise SystemExit(
            f"No '{DATA_DIR}/' folder found. Create {DATA_DIR}/healthy/ and "
            f"{DATA_DIR}/mange/ and put images in them first."
        )

    train_tf, eval_tf = build_transforms()
    train_ds, val_ds, test_ds, full_train, train_idx = make_splits(DATA_DIR, train_tf, eval_tf)
    class_names = full_train.classes  # alphabetical: ['healthy', 'mange'] -> mange is class 1
    print(f"Classes: {class_names}")
    print(f"Train: {len(train_ds)}  Val: {len(val_ds)}  Test: {len(test_ds)}")

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    model = build_model(len(class_names), device)
    criterion = nn.CrossEntropyLoss(weight=class_weights(full_train, train_idx, device))
    optimizer = torch.optim.Adam(model.fc.parameters(), lr=LR)

    best_val_acc = 0.0
    for epoch in range(1, EPOCHS + 1):
        tr_loss, tr_acc = run_epoch(model, train_loader, criterion, optimizer, device, train=True)
        vl_loss, vl_acc = run_epoch(model, val_loader, criterion, optimizer, device, train=False)
        print(f"Epoch {epoch:2d}/{EPOCHS} | "
              f"train loss {tr_loss:.3f} acc {tr_acc:.0%} | "
              f"val loss {vl_loss:.3f} acc {vl_acc:.0%}")
        if vl_acc >= best_val_acc:
            best_val_acc = vl_acc
            torch.save(model.state_dict(), MODEL_OUT)

    print(f"\nBest validation accuracy: {best_val_acc:.0%}. Model saved to {MODEL_OUT}")

    # Reload best model and judge it on data it has never seen.
    model.load_state_dict(torch.load(MODEL_OUT, map_location=device))
    evaluate(model, test_loader, device, class_names)
    save_mistakes(model, test_ds, device, class_names)


if __name__ == "__main__":
    main()
