# MITOS-ATYPIA-14_challenge
contest using breast cancer histological images

# Dataset

 Aperio and Hamamatsu are two different whole-slide image (WSI) scanner brands used to digitize the breast cancer tissue slides in this dataset:

Aperio (now Leica Biosystems) — a widely used digital pathology scanner. Produces .svs files (though here the frames are pre-extracted as TIFFs/JPEGs).
Hamamatsu (NanoZoomer) — another leading WSI scanner brand. Produces .ndpi files.
The MITOS-ATYPIA-14 challenge specifically provided the same tissue slides scanned by both scanners to test whether mitosis/atypia detection algorithms generalize across different imaging hardware. The A03/A04... slides are Aperio scans, and H03/H04... are the corresponding Hamamatsu scans of the same tissue blocks.

This is why the dataset is split that way — it's a cross-scanner generalization benchmark.  
[Download Data](https://mitos-atypia-14.grand-challenge.org/Dataset/)  
```
data/extracted/
├── training/
│   ├── aperio/    A03 A04 A05 A07 A10 A11 A12 A14 A15 A17 A18  (11 slides)
│   └── hamamatsu/ H03 H04 H05 H07 H10 H11 H12 H14 H15 H17 H18  (11 slides)
└── testing/
    ├── aperio/    A06 A08 A09 A13 A16  (5 slides)
    └── hamamatsu/ H06 H08 H09 H13 H16  (5 slides)

```
Each slide folder (A03/, H03/, etc.) contains:

|Subfolder|	Contents|
| -------- | -------- |
|frames/x10, x20, x40|	TIFF image tiles at 3 magnifications|
|mitosis/	|JPG patches + CSV annotations (mitosis / not-mitosis)|
|atypia/x20, x40	|CSV per-field nuclear atypia scores|
22 slides total — 11 aperio + 11 hamamatsu for training, 5 + 5 for testing.

The slides are stained with standard hematoxylin and eosin (H&E) dyes and they have been scanned by two slide scanners: Aperio Scanscope XT and Hamamatsu Nanozoomer 2.0-HT.
In each slide, the pathologists selected several frames at X20 magnification located inside tumours. These X20 frames are used for scoring nuclear atypia. The X20 frames have been subdivided into four frames at X40 magnification. The X40 frames are used to annotate mitosis and to give a score to six criteria related to nuclear atypia. 

# Background

### Nuclear atypia
Nuclear atypia describes abnormal, enlarged, or irregular cell nuclei, often identified in pathology reports. It signifies that cells look unusual, frequently suggesting malignancy, pre-cancerous changes, or damage from inflammation/radiation. Key features include large size (nucleomegaly), irregular membranes, and dark-staining, clumped chromatin. 

Key Features of Nuclear Atypia:
* Nuclear enlargement: Nuclei are significantly larger than normal.
* Hyperchromasia: Nuclei appear darker due to increased, dense chromatin.
* Irregular contours: The nuclear membrane is not smooth or round.
* Prominent nucleoli: Structures inside the nucleus are large and visible.
* Irregular chromatin distribution: DNA is irregularly distributed, often appearing clumped or coarse. 

Criteria that may help in scoring nuclear atypia:
1. size of nuclei, 
2. size of nucleoli, 
3. density of chromatin, 
4. thickness of nuclear membrane, 
5. regularity of nuclear contour, 
6. anisonucleosis (size variation within a population of nuclei). 
Contestants are free to use one or several of these criteria, or to create their own set of criteria, for the task of nuclear atypia scoring.



### Mitosis
Mitosis in breast cancer images refers to the visual identification of cells actively dividing (mitotic figures) within stained tumor tissue samples. Pathologists use these images to count dividing cells—which appear as dense, dark, and often uniquely shaped nuclei (e.g., prophase, metaphase)—to determine the tumor's aggressiveness, grade, and the patient's prognosis.

* Appearance: Mitotic cells appear dark purple or blue on Hematoxylin and Eosin (H&E) stained slides, often showing, condensed chromosomes, star-like, or "broken egg" shapes, distinct from resting tumor cells.
* Significance: A high count of mitotic figures (high mitosis rate) signifies fast-growing, aggressive tumors, directly impacting the tumor grade (e.g., Nottingham grading system).
* Detection Method: Traditionally done by pathologists under a microscope, this process is increasingly automated using deep learning models to identify high-risk patients.
* Measurement: The Mitotic Activity Index (MAI) is often calculated by counting mitotic figures in specific high-power fields (1.6 mm^2 or 10 fields).

# Goals

___Metrics for Nuclear Atypia:___
The goal is to give the correct nuclear atypia score (1, 2 or 3)  for nuclear pleomorphism on each frame at X20 magnification. Nuclear atypia refers to nuclei shape variations as compared to normal epithelial nuclei. The more advanced is the cancer, the more nuclei become atypical in their shape, size, internal organisation.

__Nuclear atypia score__  
|value | meaning |
| - | ----- |
| 1 | low|
| 2 | moderate|
| 3 | strong|

___Metrics for Detection of Mitosis:___  
The goal is to give the list of all mitosis visible on each frame at X40 magnification. 


## Evaluation Metrics

___Nuclear Atypia___  

|number of points | condition|
| -------- | -------- |
| -1 | An incorrect score & abs(proposed score - ground truth score) == 2|
| 0 | An incorrect score & abs(proposed score - ground truth score) == 1|
| 1 | A correct score|

___Detection of Mitosis___  
A candidate mitosis would be accepted as correctly detected if the coordinates of its centroid are within a range of 8 µm from the centroid of a ground truth mitosis.

__The metrics:__  
* D = number of mitosis detected (centroid within a range of 8 µm from the centroid of a ground truth mitosis)  
* TP = number of True Positives, that is the number of mitosis that are ground truth mitosis among the D mitosis detected
* FP = number of False Positives, that is the number of mitosis that are not ground truth mitosis among the D mitosis detected
* FN = number of False Negatives, that is the number of ground truth mitosis that have not been detected
* recall (sensitivity) = TP / (TP+FN)
* precision (positive predictive value) = TP / (TP+FP)
* F-measure = 2 * (precision * sensitivity) / (precision + sensitivity)

The rank will be given according to the F-measure.


## Setup

___Hardware summary:___

* GPU: RTX 3060 Ti
* VRAM: 8 GB
* CUDA driver: 13.1 (latest)

___Requirements___

--extra-index-url https://download.pytorch.org/whl/cu124
torch==2.6.0+cu124
torchvision==0.21.0+cu124
numpy
matplotlib
Pillow
albumentations
tqdm


## Code

|File|	Purpose|
|---------|---------|
|stain_norm.py|	Pure-numpy Macenko H&E normalizer. fit() on a reference image per scanner, transform() at runtime, save()/load() for caching|
|augmentation.py|	Two independent albumentations pipelines — get_atypia_augmentation() and get_mitosis_augmentation() — each parameterized by magnification and split. Mitosis uses KeypointParams so centroids are transformed with the image|
|splits.py|	Block-stratified K-fold (get_kfold_splits), leave-one-out, and fixed split. A03+H03 always land in the same fold — no leakage|
|dataset.py|	AtypiaDataset (48 frames from A03+H03, score 1/2/3) and MitosisDataset (192 X40 frames with centroid lists). Shared IO helpers: read_atypia_label, read_mitosis_csv, load_image_rgb|
|preview.py|	save_preview_grid() writes PNG grids of RAW / NORM / AUG columns, one file per (task × scanner). Run with "python -m CommonRoutines.preview"|


# Pipelines

## Atypia 

1. Backbone — EfficientNet-B3 (pretrained ImageNet)  
Fits comfortably in 8 GB at 512×512, batch size 12–16  
Better accuracy/parameter ratio than ResNet for medical imaging  
Strong transfer learning from ImageNet (colors, textures generalize well to H&E)  
Avoid ViT — needs more data and more VRAM to fine-tune well  
2. Head — Ordinal Regression (not plain softmax)  
Atypia score is ordinal: 1 < 2 < 3 (being wrong by 2 is penalized harder)  
Use CORN loss or a simple cumulative logit head instead of regular cross-entropy  
If you want simpler: weighted cross-entropy with penalty weight [1, 1, 2] for off-by-2  
3. Training strategy  

|Setting |	Value|
|---------|------------|
|Input size	|512×512|
|Batch size	|12–16|
|Optimizer	|AdamW, lr=1e-4|
|Scheduler	|CosineAnnealingLR|
|Epochs	|40–60|
|Early stopping	|patience=10 on val score|
|Freeze backbone	|First 5 epochs (train head only), then unfreeze all|
4. Class imbalance handling
* Check label distribution — atypia scores are often imbalanced (more 2s than 1s or 3s)
* Use weighted sampler in DataLoader so each batch sees balanced classes
* Or use weighted loss per class
5. Scanner robustness
* Stain normalize per scanner (Macenko, already built)
* Always train with both scanners mixed
* Track val metrics separately per scanner (Aperio vs Hamamatsu)  
6. Regularization (critical for small data)
* Dropout 0.3 before final FC
* Label smoothing 0.1
* Heavy augmentation (already built)
* Weight decay 1e-4 in AdamW  
7. Final model selection
Save best checkpoint by challenge metric (not loss): weighted accuracy with −1 penalty for off-by-2
At end: retrain on all 11 training blocks with best hyperparams, submit on official test set

__Givven data and GPU constraints, the design skips:__
* ViT/DINO foundation models — VRAM-heavy, need more data to beat EfficientNet here
* MIL — designed for slide-level aggregation, not frame-level scoring
* Ensemble of 5 folds — optional, only if you want max score at submission time

```
Main Flow (train.py)

main()
  ├─ get_kfold_splits(5)  → 5 (train, val) pairs
  └─ for each fold:
      ├─ load_stain_normalizers()
      ├─ create_model(EfficientNet-B3)
      ├─ training loop (40–60 epochs):
      │   ├─ train_epoch()  → forward, CORN loss, backward
      │   ├─ validate()     → ordinal logits → predictions → challenge_score
      │   └─ early stopping on challenge_score
      └─ save checkpoint
```