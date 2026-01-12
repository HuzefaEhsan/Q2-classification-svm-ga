# Q2 — Linear Classification with Outliers (SVM + Genetic Algorithm_toggle)

## Contents

- `Q2_Classification_SVM_GA.ipynb` — main notebook
- `classification_solution.py` — runnable script version
- `DataKlasifikasi(Sheet1).csv` — dataset (place in same folder)
- `outputs/` — generated figures

## Environment

- Python 3
- Dependencies in `requirements.txt`

## How to run (Anaconda)

1. Open Anaconda Prompt
2. Go to the repo folder:
   cd path/to/this/repo
3. Create and activate env (example):
   conda create -n optproj python=3.11 -y
   conda activate optproj
4. Install deps:
   pip install -r requirements.txt
5. Launch Jupyter:
   jupyter notebook
6. Open `Q2_Classification_SVM_GA.ipynb` and run all cells.

## Output

Figures will be saved to `outputs/`:

- fig1_scatter.png
- fig2_svm_boundary.png
- fig3_ga_boundary.png

## Reproducibility

- Random seed is fixed in code (SEED=42).
