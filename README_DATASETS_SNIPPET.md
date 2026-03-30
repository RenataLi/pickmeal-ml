## Datasets

### 1. Roboflow Menu Text Box v3
This is the main supervised dataset for the CV part of the project.

It is used for:
- menu text region detection;
- layout pre-segmentation before OCR;
- the first CV baseline.

Project role:
- **main annotated dataset**.

### 2. Kaggle Indian Restaurant Menu Card Images
This is an auxiliary raw image corpus.

It is used for:
- OCR stress testing;
- robustness checks;
- manual annotation of a small gold set;
- domain EDA.

Project role:
- **auxiliary unlabeled corpus**.

### 3. USDA FoodData Central
This is the main nutrition knowledge base.

It is used for:
- ingredient-level calorie lookup;
- macro and micro nutrient lookup;
- later menu enrichment.

Project role:
- **main nutrition source**.

### 4. Open Food Facts
This is a supplementary open food database.

It is used for:
- ingredient normalization;
- weak supervision;
- extra nutrition hints.

Project role:
- **supplementary knowledge source**.

## EDA
Main notebook:
- `notebooks/03_checkpoint3_dataset_and_eda.ipynb`

Generated files:
- `data/interim/roboflow_manifest.csv`
- `data/interim/roboflow_boxes.csv`
- `data/interim/kaggle_manifest.csv`
- `data/interim/quarantine_empty_labels.csv`
- `data/interim/quarantine_corrupted_images.csv`
- `data/interim/quarantine_near_full_boxes.csv`
- `data/interim/dataset_summary.json`

## Key decisions
- Roboflow is the main annotated dataset for CV and layout tasks.
- Kaggle menu images are not used as supervised ground truth.
- Nutrition and allergen enrichment are separated from CV data.
- The main MVP pipeline is: **OCR -> structured parsing -> embeddings + rules -> optional LLM repair/explanation**.
