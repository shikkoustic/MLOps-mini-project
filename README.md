# MP128 --- Tomorrow's Air Quality (Patna)

SCSE3040 MLOps Mini Project

- **Group id:** G107
- **Problem id:** MP128
- **Faculty:** Dr. Pulkit Dwivedi (Batch 9)

## Members

| Enrolment no. | Name |
|---|---|
| S24CSEU0962 | Shikhar Srivastava |
| S24CSEU0939 | Shreya Singh |
| S24CSEU0959 | Ashutosh Tiwari |

## Problem

Predict `pm25_next_day` (tomorrow's average PM2.5, micrograms per cubic
metre) for the Patna Clean Air Cell, one day ahead, so schools/hospitals can
plan. Regression problem, scored on MAE. Full spec in `PROBLEM.md`.

## Repository layout

```
src/mp128/            data-quality checks and (from Stage 2) training/cleaning code
reports/findings.yaml  the filled-in findings form
data/raw/              original CSVs, tracked with DVC
tests/                 pytest tests
app/                   FastAPI service (Stage 3)
k8s/                   Kubernetes manifests (Stage 3)
models/                champion model artifact (Stage 2+)
predictions/           final holdout predictions (Stage 4)
```

## Progress by stage

- [x] **Stage 1** --- repo + DVC set up, the seven data problems identified
      and recorded in `reports/findings.yaml` (see `src/mp128/data_checks.py`)
- [ ] **Stage 2** --- cleaning, MLflow tracking, model registry
- [ ] **Stage 3** --- FastAPI service, Docker, CI, Kubernetes manifests
- [ ] **Stage 4** --- drift monitoring, retraining, holdout predictions

## Setup

```bash
pip install -r requirements.txt
```

There is **no DVC remote** configured (not needed for grading --- the tool
only checks that `data/raw/train.csv.dvc` / `later.csv.dvc` hash matches the
original zip). After cloning, get the original data into place yourself:

```bash
unzip /path/to/G107_MP128.zip -d /tmp/mp128
cp /tmp/mp128/G107_MP128/data/train.csv data/raw/train.csv
cp /tmp/mp128/G107_MP128/data/later.csv data/raw/later.csv
dvc status   # should say "up to date" once the hashes match
python -m src.mp128.data_checks
```
