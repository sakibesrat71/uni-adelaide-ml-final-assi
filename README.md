# MNIST: digits 1 and 7

Binary classification with a small labelled training set, followed by exploratory clustering of the test images.

## Setup

Requires Python 3.11 or later. From this directory:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

The course dataset is not included. Keep `mnist_1_7_subsets-1.npz` in the parent directory or pass its location explicitly.

## Workflow

The training stage reads only `train_data` and `train_targets`. Both classifiers use identical stratified folds. The test stage must run only after model selection is saved. Clustering is a separate unsupervised analysis and cannot change the classifier.

Development commits record completed implementation stages at their actual times.
