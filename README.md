# NSL-KDD Intrusion Detection: ANFIS vs Neural Network

## Project overview

This project compares a feature-weighted first-order Sugeno Adaptive Neuro-Fuzzy Inference System (ANFIS) with a conventional feed-forward Neural Network (MLP) for binary intrusion detection using the NSL-KDD dataset.

Each connection is classified as:

- `0`: Normal traffic
- `1`: Attack traffic (any label other than `normal`)

The official `KDDTrain+` file is used for model fitting and validation. The official `KDDTest+` file is used for final evaluation. The `difficulty` field is excluded from model inputs.

## Four experiments

The project has four-experiment comparison structure. 

| Experiment | Classifier | Original input attributes |
|---|---|---:|
| 1 | ANFIS | All 41 |
| 2 | Fisher Score + ANFIS | Top 17 |
| 3 | Neural Network (MLP) | All 41 |
| 4 | Fisher Score + Neural Network (MLP) | Top 17 |

This design compares ANFIS and MLP with the same feature sets, and separately measures the effect of Fisher Score feature selection.

## Project files

```text
project/
├── anfis_nsl_kdd.py
├── anfis_notebook.ipynb
├── README.md
└── results/                 # created automatically when the code runs
```

- `anfis_nsl_kdd.py`: complete runnable implementation.
- `anfis_notebook.ipynb`: notebook containing the same complete implementation in a code cell.
- `README.md`: setup, methodology, and output documentation.
- `results/`: generated CSV tables and plots.

## Dataset

Dataset source: [HoaNP/NSL-KDD-DataSet](https://github.com/HoaNP/NSL-KDD-DataSet)

Required files:

- `KDDTrain+.txt`
- `KDDTest+.txt`

Place both files in the project directory, or run the script with internet access so it can try the source URLs automatically. The files should be the original comma-separated text files without a header. Each row contains 41 traffic attributes, an attack label, and a difficulty value (43 columns total).

## Preprocessing

- `protocol_type`, `service`, and `flag` are one-hot encoded.
- The remaining numeric attributes are standardised using statistics fitted only on the fitting partition.
- Unknown categories are handled by the one-hot encoder.
- The full-feature path uses 41 original attributes; the Fisher path uses the top 17 original attributes. Encoded input-column counts can be larger because categorical attributes expand into multiple indicator columns.
- Fisher ranking is computed on fitting data only. Validation and official test labels are not used to rank features.

## Data split and threshold policy

The training file is divided into three disjoint partitions:

1. Model-fitting partition: fits preprocessing and model parameters.
2. Model-selection validation partition: selects model configurations and monitors ANFIS training.
3. Threshold-calibration validation partition: selects the final classification threshold.

The validation partitions hold out selected attack families from model fitting. Attack-family names are used only to construct the split; the classifiers receive binary normal/attack targets.

The primary threshold policy selects the threshold that maximises attack recall while keeping the threshold-calibration FPR at or below 3%. This is a validation constraint, not a guarantee that the official test FPR will also be at or below 3%. Test FPR is measured independently.

## Models

### ANFIS

The ANFIS is a feature-weighted, first-order Sugeno model. Rule centres are initialised with class-aware MiniBatchKMeans clustering on fitting data. Gaussian membership widths are initialised from within-cluster variation. Each rule has a local linear Sugeno consequent. Hybrid learning estimates consequent coefficients using regularised weighted least squares and updates premise parameters using Adam. The model-selection process searches rule counts `4, 6, 8, 12, 16, 24, 32` and ridge values `0.00001, 0.0001, 0.001, 0.01`.

### Conventional Neural Network (MLP)

The baseline is `sklearn.neural_network.MLPClassifier`, a feed-forward multilayer perceptron. It uses ReLU activation and the Adam optimizer. The search evaluates eight configurations:

- Hidden layers: `(32,)`, `(64,)`, `(64, 32)`, `(128, 64)`
- L2 regularisation `alpha`: `0.0001` and `0.001`

Other settings are batch size `512`, initial learning rate `0.001`, maximum iterations `150`, early stopping enabled, validation fraction `0.10`, and `n_iter_no_change=12`. The best configuration is selected using model-selection validation F1-score, with recall and FPR as tie-breakers. The final model is refitted on the complete fitting partition. The separate threshold-calibration partition is then used to select the operating threshold.

## Evaluation metrics

All four experiments are evaluated on the same official `KDDTest+` rows. Attack is the positive class.

| Metric | Meaning |
|---|---|
| Accuracy | Fraction of all connections classified correctly |
| Precision | Fraction of predicted attacks that are actual attacks |
| Attack recall / TPR | Fraction of all attacks detected |
| Attack F1-score | Harmonic mean of attack precision and recall |
| FPR | Fraction of normal connections incorrectly flagged as attacks, `FP / (FP + TN)` |
| TN, FP, FN, TP | Confusion-matrix counts |
| Fit time | Time for the final model fit |
| Tuning time | Time spent comparing configurations |

## Generated results

Running the script regenerates the result tables and all comparison plots using the current ANFIS and NN models. 

### CSV files

- `results/four_experiment_results.csv` — main four-experiment test comparison, model settings, thresholds, timing, and confusion-matrix counts.
- `results/operating_point_analysis.csv` — F1-max threshold and thresholds selected under validation FPR caps of 1%, 2%, 3%, and 5%, with validation and test metrics.
- `results/top17_features.csv` — 17 original attributes selected by Fisher Score.

### Plots

- `results/metric_comparison.png` — grouped Accuracy, Attack F1-score, and Attack Recall comparison.
- `results/fpr_comparison.png` — test-set FPR comparison across all four experiments.
- `results/confusion_matrices.png` — four confusion matrices, one per experiment.



## Installation

Python 3.10 or newer is recommended.

```bash
python -m pip install numpy pandas scikit-learn matplotlib torch requests jupyter
```

If PyTorch is already installed, retain the installation appropriate for your CPU/GPU environment.

## Run the Python script

From the project directory:

```bash
python anfis_nsl_kdd.py
```

The script automatically uses CUDA when available to train ANFIS; otherwise it uses the CPU. The MLP baseline is trained by scikit-learn. The search can take several minutes, particularly on CPU.

## Run the notebook

```bash
jupyter notebook anfis_notebook.ipynb
```

Run the notebook cells from top to bottom. It contains the complete implementation, so it can run independently of the `.py` file as long as the dependencies and dataset are available.

## Reproducibility notes

- Random seed: `42`.
- Preprocessing is fitted on fitting data only.
- Feature ranking is fitted on fitting data only.
- Model selection, threshold calibration, and official test evaluation are kept separate.
- The 3% FPR limit applies to the threshold-calibration partition, not necessarily to the final test set.
- Model results can vary slightly across library versions and hardware.
