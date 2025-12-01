from pathlib import Path
import json, os, itertools
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
import imagehash
import cv2
from skimage.metrics import structural_similarity as ssim


DATA_ROOT = Path("/content/drive/MyDrive/brainMri")   # original dataset root with train/ and test/
OUT_DIR   = Path("/content/drive/MyDrive/mavic_project/results")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# thresholds
HAMMING_MAX = 6       # <=6 works well for 256px pHash/dHash
SSIM_MIN    = 0.92    # only accept pairs with SSIM >= 0.92
IMG_SIZE    = 256     # resize for hashing/ssim
def list_images(root: Path, splits=('train','test')):
    records = []
    for split in splits:
        base = root / split
        if not base.exists():
            continue
        for cls in sorted(os.listdir(base)):
            cdir = base / cls
            if not cdir.is_dir():
                continue
            for fname in os.listdir(cdir):
                if fname.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')):
                    records.append({
                        "path": str(cdir / fname),
                        "split": split,
                        "cls": cls,
                        "fname": fname
                    })
    return pd.DataFrame(records)

def load_for_hash(path, size=IMG_SIZE):
    img = Image.open(path).convert('RGB')
    img = img.resize((size, size), Image.BILINEAR)
    return img

def load_for_ssim(path, size=IMG_SIZE):
    im = cv2.imread(path, cv2.IMREAD_COLOR)
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    im = cv2.resize(im, (size, size), interpolation=cv2.INTER_AREA)
    return im

def compute_hashes(pil_img):
    # three complementary hashes
    ph = imagehash.phash(pil_img)   # robust to small changes
    dh = imagehash.dhash(pil_img)
    ah = imagehash.average_hash(pil_img)
    return ph, dh, ah

def hamming(a, b):  # imagehash distance
    return a - b

def ssim_score(a_rgb, b_rgb):
    # SSIM on luminance to reduce color effects
    a_gray = cv2.cvtColor(a_rgb, cv2.COLOR_RGB2GRAY)
    b_gray = cv2.cvtColor(b_rgb, cv2.COLOR_RGB2GRAY)
    s, _ = ssim(a_gray, b_gray, full=True)
    return float(s)
  df = list_images(DATA_ROOT)
print(f"Found {len(df)} images across splits/classes")
df.head()
# compute hashes
ph_list, dh_list, ah_list = [], [], []
for p in tqdm(df["path"], desc="Hashing"):
    img = load_for_hash(p)
    ph, dh, ah = compute_hashes(img)
    ph_list.append(ph)
    dh_list.append(dh)
    ah_list.append(ah)

df["phash"] = ph_list
df["dhash"] = dh_list
df["ahash"] = ah_list
# Build simple buckets by pHash string to reduce comparisons
from collections import defaultdict

bucket = defaultdict(list)
for idx, row in df.iterrows():
    # coarse key to group likely matches; string prefix helps reduce pairs
    key = str(row["phash"])[:8]
    bucket[key].append(idx)

pairs = []
for key, idxs in tqdm(bucket.items(), desc="Scanning buckets"):
    if len(idxs) < 2:
        continue
    # check all pairs inside bucket
    for i, j in itertools.combinations(idxs, 2):
        a = df.loc[i]; b = df.loc[j]
        # Type-C: different splits (e.g., train vs test)
        if a["split"] == b["split"]:
            continue

        # quick multi-hash gate
        ham = (hamming(a["phash"], b["phash"]) +
               hamming(a["dhash"], b["dhash"]) +
               hamming(a["ahash"], b["ahash"]))
        if ham > 3*HAMMING_MAX:
            continue

        # SSIM verification
        a_img = load_for_ssim(a["path"])
        b_img = load_for_ssim(b["path"])
        s = ssim_score(a_img, b_img)
        if s >= SSIM_MIN:
            pairs.append({
                "a_path": a["path"], "a_split": a["split"], "a_cls": a["cls"],
                "b_path": b["path"], "b_split": b["split"], "b_cls": b["cls"],
                "ham_sum": int(ham), "ssim": round(s, 4),
                "type": "C"  # cross-split
            })

len(pairs)
rep_df = pd.DataFrame(pairs).sort_values(by=["ssim","ham_sum"], ascending=[False, True])
csv_path  = OUT_DIR / "near_duplicates_typeC_report.csv"
json_path = OUT_DIR / "near_duplicates_typeC_report.json"
rep_df.to_csv(csv_path, index=False)
with open(json_path, "w") as f:
    json.dump(pairs, f, indent=2)
print("Saved:", csv_path, "\nSaved:", json_path)

# Create a "to_remove.txt" proposal:
# policy: keep TRAIN; drop the TEST side if cross-split
to_remove = []
for r in pairs:
    if r["a_split"] == "test" and r["b_split"] == "train":
        to_remove.append(r["a_path"])
    elif r["b_split"] == "test" and r["a_split"] == "train":
        to_remove.append(r["b_path"])
    else:
        # if both are in different custom splits, keep first, mark second
        to_remove.append(r["b_path"])

to_remove = sorted(set(to_remove))
rem_path = OUT_DIR / "near_duplicates_typeC_to_remove.txt"
with open(rem_path, "w") as f:
    f.write("\n".join(to_remove))
print("Saved removal proposal:", rem_path, f"\nCount: {len(to_remove)}")
