import argparse
from pathlib import Path
import torch
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.preprocessing import label_binarize
import seaborn as sns
import matplotlib.pyplot as plt

from mavic_ptnet.models.mavic_ptnet import MaViCNetPT
from mavic_ptnet.data.datasets import MultiViewDataset, CLASSES

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_root", type=str, required=True)
    p.add_argument("--remove_list", type=str, required=True)
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--out_dir", type=str, required=True)
    p.add_argument("--views", type=int, default=3)
    p.add_argument("--img_size", type=int, default=256)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--num_workers", type=int, default=2)
    return p.parse_args()

def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # load removal list
    remove_set = set()
    for line in open(args.remove_list):
        path = line.strip()
        if path:
            remove_set.add(path)

    test_ds = MultiViewDataset(
        root=Path(args.data_root),
        split="test",
        classes=CLASSES,
        img_size=args.img_size,
        num_views=args.views,
        exclude_set=remove_set
    )
    test_loader = DataLoader(
        test_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True
    )

    model = MaViCNetPT(num_classes=len(CLASSES), d_model=256,
                       backbone="convnext_base", head_dropout=0.20, views=args.views)
    state = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(state)
    model = model.to(device).eval()

    all_probs, all_y = [], []
    with torch.no_grad():
        for mv, y, _ in test_loader:
            mv = mv.to(device, non_blocking=True)
            logits = model(mv)
            probs = torch.softmax(logits, dim=1)
            all_probs.append(probs.cpu().numpy())
            all_y.append(y.numpy())

    probs = np.concatenate(all_probs, 0)
    y_true = np.concatenate(all_y, 0)
    y_pred = probs.argmax(1)

    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro")

    y_true_oh = label_binarize(y_true, classes=list(range(len(CLASSES))))
    try:
        macro_auc = roc_auc_score(y_true_oh, probs, average="macro", multi_class="ovr")
    except ValueError:
        macro_auc = float("nan")

    print(f"[CLEAN-TEST] MaViC-PTNet → acc {acc:.4f} | f1 {macro_f1:.4f} | auc {macro_auc:.4f}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(CLASSES))))
    plt.figure(figsize=(5.4, 4.8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="YlGnBu", cbar=True,
                linewidths=0.6, linecolor="white",
                annot_kws={"size": 12, "color": "black"},
                xticklabels=CLASSES, yticklabels=CLASSES)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix — Clean Test")
    plt.tight_layout()
    plt.savefig(out_dir / "confusion_matrix_clean_test.png", dpi=300)

if __name__ == "__main__":
    main()
