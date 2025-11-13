# MaViC-PTNet: Multi-View ConvNeXt-based Brain Tumor Classification

This repository contains the implementation of MaViC-PTNet, a multi-view architecture for four-class brain tumor classification on T1-weighted MRI images.

## 1. Dataset

We use the public Kaggle dataset:

- Brain Tumor MRI Dataset  
  https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset

Expected structure:

```text
data/
└── brainMri/
    ├── train/
    │   ├── glioma/
    │   ├── meningioma/
    │   ├── notumor/
    │   └── pituitary/
    └── test/
        ├── glioma/
        ├── meningioma/
        ├── notumor/
        └── pituitary/

2. Trained checkpoint

The trained MaViC-PTNet checkpoint used in the paper is available on Google Drive:

mavic_pt_best.pt:
https://drive.google.com/file/d/1lM4LiMG223h0qHjsran42KDIdJdmM019/view?usp=sharing

Download this file and place it under:

checkpoints/mavic_pt_best.pt

3. Environment
pip install -r requirements.txt

4. Duplicate removal

We detect cross-split near-duplicate images with perceptual hashing and SSIM and remove the test-side image when a near-duplicate pair spans train and test. The resulting removal list is stored as:

data/cleaned_indices/near_duplicates_typeC_to_remove.txt

5. Evaluation on the cleaned test set
python scripts/eval_mavic_pt_clean_test.py \
  --data_root data/brainMri \
  --remove_list data/cleaned_indices/near_duplicates_typeC_to_remove.txt \
  --checkpoint checkpoints/mavic_pt_best.pt \
  --out_dir results/


This will compute accuracy, macro-F1, macro-AUC and save the confusion matrix for the cleaned test set.

