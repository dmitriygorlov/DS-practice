# Reproducible notebooks — 2026-11-18

The original notebooks remain teaching experiments. Their saved outputs and Russian
conclusions are historical; they are **not** results from this environment. New
executions are written under ignored `artifacts/`. No performance speedup is claimed.

## Environment and migration

The original repository had no Python/dependency lock and neither notebook recorded
a Python version. Its PyCaret calls used the old `silent`, `enable_colab` and `Label`
API. Exact historical dependency versions cannot be reconstructed from this repo.

Use **Python 3.15.2**, with `requirements/core.lock` (Windows x64 validation target).
Direct versions: NumPy 2.5.5, pandas 3.1.1, SciPy 1.18.1, scikit-learn 1.9.1,
LightGBM 4.7.0, NLTK 3.10.4, matplotlib 3.11.2. `core.in` records the selected direct
dependencies; the lock records transitive versions. No Dockerfile existed; a container
was not added because these local CPU notebooks are reproducible with isolated kernels.

PyCaret 3.3.2 was the stable PyPI release checked on this date. Its NumPy `<1.27`,
pandas `<2.2`, SciPy `<=1.11.4` constraints would hold back this experiment. The used
regression operations now live in `california_models.py`: five candidate regressors,
training-only MAPE CV, tuning of the actual selected estimator, bagging, blending,
stacking, optional power transformation/scaling, prediction and persistence.
This is an explicit sklearn implementation, not a drop-in implementation of all
PyCaret APIs. Unlike the old implicit PyCaret holdout, it uses the supplied training
table for CV/refit and reserves the external test table for evaluation. Rankings and
metrics must be recomputed. Model names are taken from actual estimators, rather than
assuming a stable ranking; `rawdf` and duplicate prediction calls were fixed.

Wine regression now fits the TF-IDF vocabulary within CV folds. The lemmatized branch
uses its own pipeline rather than default arguments bound to the original sparse
matrix. Model keys are consistent tuples, tuning stores/returns the fitted winner
even with display disabled, and the selected pipeline carries its own vocabulary and
lemmatization into prediction and serialization. `max_depth` replaces ignored `depth`.
The incorrect early-stopping path using the external test set was removed. A seeded
sample and genuinely unused rows support reproducible follow-up evaluation. Model
selection still uses the notebook's evaluation split; those scores are not an unbiased
final estimate. The additional unused-row evaluation is reported separately.

Old `wineprice.model`, `count_tf_idf.model` and PyCaret pipelines have no reliable
environment/preprocessing provenance. They are not automatically loaded or converted.
Retrain from the original data. New artifacts are complete joblib pipelines; only load
trusted artifacts under the same lock. Same-environment save/load is regression-tested.

## Run

From the repository root (PowerShell):

```powershell
py -3.15 -m venv .venv
.venv/Scripts/python -m pip install -r requirements/core.lock
.venv/Scripts/python -m pip check
.venv/Scripts/python prepare_nltk.py
.venv/Scripts/python -m pytest -q
.venv/Scripts/python validate_notebooks.py
.venv/Scripts/python -m ruff check .
.venv/Scripts/python smoke_notebooks.py
```

`prepare_nltk.py` explicitly downloads public dictionaries into `data/nltk_data`.
Notebook execution itself installs no packages and downloads no datasets.
Put original `california_housing_train.csv`, `california_housing_test.csv` and
`winemag-data_first150k.csv` in `data/`, or set `DATA_DIR` to a dedicated directory.
Original Colab/Drive references are preserved in `notebook-inventory.json`.
Do not substitute private working data. For example:

```powershell
.venv/Scripts/python validate_notebooks.py google_california_pycaret.ipynb
```

Set `NOTEBOOK_SMOKE=1` for reduced CV/sample/search sizes on supplied data. The
`smoke_notebooks.py` script instead creates and deletes clearly synthetic tables and
executes both complete notebooks. Its scores are not wine/housing benchmark results.

## Verification and limits

- Both notebook schemas and all code cells are validated; Ruff checks fatal Python errors.
- Regression tests exercise both California preprocessing variants and all ensemble
  operations, prediction, sparse text CV, model-selection bookkeeping, lemmatization,
  and joblib prediction equivalence.
- CI performs dependency checks, unit tests and complete synthetic notebook execution.
- Original wine/housing datasets and old serialized models were not available locally;
  full original-data training and cross-version legacy-model loading were not validated.
- No application/server/deployment configuration was present; GitHub Pages was disabled
  at inspection. The added workflow only runs tests, with read-only repository access.
- Rollback: revert the modernization commit, recreate its predecessor's environment
  separately, and use the original historical notebooks. Baseline commit:
  `f2d3bb413519f106389e08353a19920b1811076a`.

Release/API evidence: [Python releases](https://www.python.org/downloads/source/),
[PyCaret package metadata](https://pypi.org/pypi/pycaret/3.3.2/json),
[pandas Copy-on-Write](https://pandas.pydata.org/docs/user_guide/copy_on_write.html),
[sklearn releases](https://scikit-learn.org/stable/whats_new/v1.9.html).
