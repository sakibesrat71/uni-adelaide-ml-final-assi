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

```powershell
python run.py explore
python run.py train
python run.py evaluate
$env:OMP_NUM_THREADS = "1"
$env:OPENBLAS_NUM_THREADS = "1"
python run.py cluster
python -m unittest discover -s tests -v
```

Use `--data "path/to/dataset.npz"` and `--output results` on any experiment command to change locations. Run commands from this project directory. Clustering may take several minutes. Thread limits can reduce overhead on machines with many CPU cores.

## Design choices

- Fixed division by 255 maps images to [0, 1]. Each fold separately removes constant pixels and optionally fits `StandardScaler` inside the pipeline.
- Logistic regression tests six C values and two scaling choices (12 candidates). The RBF SVM tests five C values, four gamma values, and two scaling choices (40 candidates).
- Both searches share 25 validation splits: five repeats of stratified five-fold cross-validation with seed 42. Every validation fold contains ten examples per class.
- Accuracy selects hyperparameters because the training classes are balanced and no asymmetric error cost was supplied. Balanced accuracy, F1 for digit 7, and ROC AUC provide additional views. The simpler logistic model wins an exact cross-model accuracy tie; the first grid candidate wins a within-model tie.
- Selection is saved and both final pipelines are fitted on all 100 training examples before either test array is read. Test results cannot change the winner. Training is blocked in an output directory that already contains test metrics; evaluation refuses to overwrite an existing result.
- Test labels select only actual sevens for the explicitly required unsupervised analysis. K-means first fits k=100. A separate search evaluates every k from 2 through 99 with 20 initialisations and seed 42. Maximum mean Euclidean silhouette selects k; inertia is a supporting diagnostic. PCA is used for display only.
- All reported classifier test predictions use the original fixed decision thresholds. No threshold adjustment, test-driven feature engineering, or classifier retraining follows evaluation.

Repeated validation folds overlap, so their standard deviations are descriptive and are not confidence intervals. Grid-search scores can be optimistic because the same validation results choose hyperparameters. The held-out test evaluation provides the independent comparison. Silhouette identifies a preferred partition under the chosen distance and search rule; it does not prove that handwriting has a unique true cluster count.

## Files

- `src/mnist_assignment/data.py`: validation and fixed intensity conversion.
- `explore.py`: training statistics, examples, means and distributions.
- `train.py`: pipelines, grids, shared folds and frozen selection.
- `evaluate.py`: test metrics, confusion matrices, ROC curves and error examples.
- `cluster.py`: digit-seven clustering, model selection, centres and representatives.
- `tests/test_protocol.py`: split isolation, malformed-data handling and core numerical checks.

Generated files go under `results/`: exploration JSON/figures, full cross-validation CSVs, saved folds, fitted models, test predictions/metrics, clustering scores and all cluster visualisations. Data and generated binary models are excluded from Git. Obtain the dataset through the course LMS; the repository does not redistribute course materials.

## Observed run

Training: 50 ones and 50 sevens. Both selected configurations reached mean validation accuracy 0.996. The frozen winner was logistic regression with C=1 and fixed intensity scaling. The selected SVM used C=100 and gamma=0.001, also with fixed intensity scaling. Test accuracy was 0.943 and 0.944 respectively. The test split contains 100 ones and 900 sevens; interpret raw accuracy alongside balanced accuracy and per-class results. A 0.001 test-accuracy difference is one image and does not establish a reliable advantage.

## Submission

Clustering of the 900 test sevens selected k=2 by mean silhouette (0.12174), with cluster sizes 406 and 494. The required initial k=100 partition had silhouette 0.06549. The low scores indicate overlapping handwriting styles rather than sharply separated populations.

The local submission bundle contains the Python source and one PDF report, without the dataset or fitted models. Rename its ZIP to `StudentID_FullName.zip` if personal details were not supplied. Review the analysis and follow the course's rules on assistance and attribution before submission.
