# pickmeal-ml

PickMeal ML is the machine learning part of the PickMeal project.

The goal of the project is to extract structured information from restaurant menu images and use it later for filtering, recommendation, and service development.

## Project goals

The ML pipeline focuses on these tasks:

1. detect and localize menu text regions;
2. extract OCR text from menu pages;
3. parse menu text into structured items;
4. prepare menu items for recommendation and filtering;
5. support future nutrition and allergen enrichment.

## Data sources

### Main sources
- **Roboflow Menu Text Box v3** - main annotated dataset for CV and layout tasks
- **Kaggle Indian Restaurant Menu Card Images** - auxiliary raw-domain corpus for OCR checks and parser annotation
- **USDA FoodData Central** - main nutrition knowledge base
- **Open Food Facts** - additional source for normalization and weak supervision

## Completed stages

## Stage 1 - Dataset audit and EDA
At this stage, the repository was prepared and the available datasets were audited.

Completed work:
- checked raw file structure;
- created dataset manifests;
- detected corrupted files;
- detected empty or missing labels;
- checked image sizes and basic quality;
- inspected annotation quality and edge cases;
- fixed the final role of each data source.

Main decision:
- Roboflow is used as the main supervised dataset for CV/layout;
- Kaggle menu pages are used as a raw-domain corpus for OCR and parser evaluation;
- nutrition sources are separated from image datasets.

## Stage 2 - Annotation pipeline for parser evaluation
A parser gold set was created through an annotation workflow.

Completed work:
- created annotation batches;
- prepared support files for annotation;
- used LLM-assisted annotation for menu items;
- reviewed uncertain rows;
- finalized batch-level gold files;
- rebuilt OCR files for annotation batches.

## Stage 3 - Gold dataset assembly
The annotation batches were merged into one parser dataset.

Completed work:
- merged reviewed annotation batches;
- removed duplicates;
- built a unified menu-level JSONL file;
- split the dataset by `menu_id` into train, validation, and test sets.

Current parser dataset:
- 39 menus
- 1012 item rows after merge and dedup
- split by menu_id:
  - train: 24 menus
  - valid: 7 menus
  - test: 8 menus

## Stage 4 - Parser baseline
A first rule-based parser baseline was evaluated on the menu-level gold set.

### Validation results
- item precision: 0.3037
- item recall: 0.3009
- item F1: 0.3023
- section accuracy: 0.0615
- price accuracy: 0.4074
- description exact match: 0.0000

### Test results
- item precision: 0.2955
- item recall: 0.3318
- item F1: 0.3126
- section accuracy: 0.0000
- price accuracy: 0.3333
- description exact match: 0.0000

## Main findings so far

The parser baseline is working, but it is still weak.

The main current error sources are:
- missed menu items;
- false positive items;
- empty predicted names;
- wrong section assignment;
- wrong price extraction;
- weak multiline description grouping.

This means that the current bottleneck is not data collection alone.  
The main next step is to improve parser logic.

## Repository structure

```text
src/pickmeal_ml/data/
    build_manifests.py
    make_annotation_batch.py
    finalize_llm_annotation.py
    rebuild_existing_batch_ocr.py
    merge_annotation_batches.py
    build_full_gold_jsonl.py
    split_full_gold_by_menu_id.py
    make_clean_detection_dataset.py

src/pickmeal_ml/models/
    menu_structuring_baseline.py
    evaluate_parser_full.py
    hybrid_recommender_baseline.py
    train_yolo_baseline.py

notebooks/
    01_eda_menu_datasets.ipynb
    03_checkpoint3_dataset_and_eda.ipynb

reports/parser_baseline/
    parser_metrics_valid.json
    parser_metrics_test.json
    parser_errors_valid.csv
    parser_errors_test.csv