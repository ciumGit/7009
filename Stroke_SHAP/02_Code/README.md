# Stroke classification with SHAP explanations

train a stroke classifier on the Kaggle Stroke Prediction
Dataset and explain it with SHAP at local and global level, for three
audiences, plus a second black-box model explained with KernelSHAP.

## Layout

```
data/healthcare-dataset-stroke-data.csv   dataset (as shipped in the course repo)
src/prep.py                   stratified 60/20/20 split + leak-free preprocessing
src/01_task1_eda.py           task 1 - EDA panel (figure 1) + eda_stats.json
src/02_task2_model_shap.py    task 2 - XGBoost + SHAP TreeExplainer + additivity check
src/03_task3_local_force.py   task 3 - local force plot of one held-out case (figure 2)
src/04_task4_global_summary.py task 4 - global summary plots (figure 3a-3c)
src/06_task6_alt_model.py     task 6 - RBF SVM explained with KernelSHAP (figures 4-5)
outputs/                      metrics (json), SHAP arrays (npy), figures (png),
                              fitted XGBoost model (joblib), test-split rows
logs/eda_baseline.log         unmodified run of the course EDA script
```

## Run order

```bash
python -m venv venv && venv/bin/pip install -r requirements.txt
venv/bin/python src/01_task1_eda.py            #  3 s
venv/bin/python src/02_task2_model_shap.py     #  6 s (includes a 12-point grid search)
venv/bin/python src/03_task3_local_force.py    #  4 s
venv/bin/python src/04_task4_global_summary.py #  3 s
venv/bin/python src/06_task6_alt_model.py      # 80 s, 69 s of it KernelSHAP
```

Wall-clock times on the machine used for the reported run; 06 runs before 05
because the slide deck quotes the SVM's numbers and the report reads every
metrics file.

## Seeds and data

All randomness fixed with seed 42 (split, model, background sample, KernelSHAP
subsample). Dataset SHA-256 in `outputs/data_sha256.txt`. The test split is
evaluated once, after hyper-parameters and the decision threshold are frozen on
validation.

## Departures from the pipeline 

1. Preprocessing fitted on the training split only (the course pipeline encodes
   and imputes before splitting).
2. Missing BMI imputed with the training median, not 0 kg/m2.
3. Stratified split; the course split is unstratified.
4. Class weighting instead of random over-sampling.
5. Figures saved to disk instead of plt.show().
