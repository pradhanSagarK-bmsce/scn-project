# Intrusion Detection Using Fisher Score-Based Feature Selection and ANFIS

## Project overview

This project builds and compares four binary intrusion-detection systems using the NSL-KDD network traffic dataset. The main objective is to study whether a feature-weighted Sugeno Adaptive Neuro-Fuzzy Inference System (ANFIS) can detect malicious traffic effectively, and how Fisher Score-based feature selection affects its performance. A Decision Tree classifier is used as a conventional machine-learning baseline.

Each network connection is classified as one of two classes:

- **Normal (`0`)**: the connection is labelled `normal` in NSL-KDD.
- **Attack (`1`)**: the connection has any other attack label.

The project uses the official `KDDTrain+` file for model development and `KDDTest+` for evaluation. The `difficulty` column is excluded from the model inputs. The implementation includes data preprocessing, feature ranking, model selection, threshold calibration, evaluation metrics, result tables, and comparison plots.

The purpose is not only to compare accuracy. Intrusion detection must also consider how many attacks are caught, how many attacks are missed, and how often normal traffic is incorrectly flagged.

## Project files

The project directory is organised as follows:

```text
scn-project/
├── anfis_notebook.ipynb
├── anfis_nsl_kdd.py
├── README.md
└── results/
```

- **`anfis_notebook.ipynb`** contains the notebook workflow for running the experiments and viewing their results.
- **`anfis_nsl_kdd.py`** contains the Python implementation of preprocessing, feature selection, ANFIS and Decision Tree training, threshold calibration, evaluation, and plotting.
- **`README.md`** documents the purpose, setup, experiment design, and results.
- **`results/`** is created automatically when the code runs. It contains the report-ready CSV tables and generated plots.

## Dataset

The project uses the NSL-KDD dataset, a labelled dataset for network intrusion detection.

**Dataset source:** [HoaNP/NSL-KDD-DataSet on GitHub](https://github.com/HoaNP/NSL-KDD-DataSet)

The script first looks for local copies of the dataset files. It checks the current working directory and the directory containing the Python script. If the files are not found locally, it tries the source URLs defined in the code.

The expected files are:

- `KDDTrain+.txt`
- `KDDTest+.txt`

Place the files in the `scn-project/` directory if downloading them automatically is not possible. The files should be the original comma-separated NSL-KDD text files without a header row.

Each row contains 41 traffic attributes, an attack label, and a difficulty-level value. The script assigns column names when loading the files. It checks that the input file has the expected 43 columns.

### The 41 traffic attributes

The original NSL-KDD attributes describe several aspects of a connection. The following groups provide a practical overview; the exact column names used by the implementation are retained in `anfis_nsl_kdd.py`.

**Basic connection information** includes connection duration, protocol type, service, connection flag, source and destination byte counts, land status, wrong-fragment count, and urgent-packet count.

**Content and login information** includes indicators or counts associated with suspicious content, failed logins, whether the user is logged in, compromised conditions, root-shell activity, root access, file creation, shell use, file access, outbound commands, host login, and guest login.

**Recent-traffic statistics** include connection counts, service-specific counts, error rates, same-service rate, different-service rate, and service diversity across destination hosts.

**Destination-host statistics** describe the number of recent connections to a host or service and rates associated with service similarity, source ports, host diversity, and connection errors.

The full-feature experiments use all 41 original attributes. The feature-selection experiments use the 17 highest-ranked original attributes according to Fisher Score. The label is used as the target, and the difficulty value is not provided to either classifier.

## Software requirements

The implementation uses Python and the following libraries:

- NumPy for numerical calculations
- pandas for loading and handling tabular data
- scikit-learn for preprocessing, clustering, metrics, and Decision Tree classification
- PyTorch for the trainable ANFIS model
- Matplotlib for plots
- requests for downloading dataset files when needed

From the project directory, install the dependencies with:

```bash
python -m pip install numpy pandas scikit-learn matplotlib torch requests
```

A Python environment that supports the type annotations used in the script is recommended. If PyTorch is already installed, use the installation appropriate for your environment rather than reinstalling it unnecessarily.

## Running the project

### Option 1: Run the notebook

1. Open `anfis_notebook.ipynb` in Jupyter Notebook, JupyterLab, or Google Colab.
2. Make sure the required libraries are available.
3. Make sure `KDDTrain+.txt` and `KDDTest+.txt` can be found locally, or allow the script's dataset download step to access the source repository.
4. Run the notebook cells in order.
5. Review the final comparison table and the files created in `results/`.

### Option 2: Run the Python script

From a terminal, move to the project directory and execute:

```bash
cd /scn-project
python anfis_nsl_kdd.py
```

Adjust the `cd` path if the directory is located elsewhere on your computer. The script automatically chooses a CUDA-enabled GPU when PyTorch reports one as available; otherwise, it runs on the CPU.

The fitting process and hyperparameter search can take longer on a CPU than on a GPU. The output includes progress messages for data loading, the four experiments, training, threshold calibration, and evaluation.

## Workflow at a glance

The pipeline follows these stages:

1. Load `KDDTrain+` and `KDDTest+`.
2. Convert the attack labels into binary targets: normal = `0`, attack = `1`.
3. Create separate fitting, model-selection, and threshold-calibration subsets from `KDDTrain+`.
4. Fit preprocessing on the fitting data and apply it to the validation and test data.
5. Rank original attributes with Fisher Score and select the top 17 for the reduced-feature experiments.
6. Train and tune ANFIS and Decision Tree models on the full and reduced feature sets.
7. Choose a decision threshold from the separate threshold-calibration subset.
8. Evaluate the selected models on `KDDTest+`.
9. Save the results tables and plots in `results/`.

## Data preparation

### Binary target labels

The label is stripped of surrounding whitespace and a trailing period, then converted to lowercase. A row is labelled `0` only when the label is `normal`; every other label becomes `1`.

This setup allows the project to focus on the first question an intrusion-detection system must answer: **is the connection normal or malicious?** It does not attempt to identify the specific attack family at prediction time.

### Numeric and categorical inputs

Three original attributes are categorical:

- `protocol_type`
- `service`
- `flag`

These attributes are one-hot encoded. This creates a separate indicator column for each category learned from the fitting data. Categories not encountered during fitting are handled by the encoder without causing a transformation error.

The other attributes are treated as numeric and standardised using statistics learned from the fitting subset. The same fitted preprocessing steps are applied to the model-selection validation data, threshold-calibration data, and official test data. Preprocessing is not fitted separately on the test set.

In the supplied results, the full 41-attribute path produces **116 encoded columns**, while the selected 17-attribute path produces **92 encoded columns**. The encoded column count differs from the number of original attributes because categorical attributes expand into multiple one-hot columns.

## Data splitting and validation

The code separates model fitting, model selection, threshold calibration, and final test evaluation.

| Partition | Rows in the reported run | Purpose |
|---|---:|---|
| Model fitting | 106,463 | Fit preprocessing and model parameters |
| Model-selection validation | 8,412 | Compare model settings and monitor ANFIS training |
| Threshold-calibration validation | 11,098 | Select the classification threshold |
| Official `KDDTest+` | 22,544 | Calculate final evaluation metrics |

The two validation subsets are drawn from `KDDTrain+` and are disjoint. The code holds selected attack types out of the fitting subset. For the reported run, the model-selection subset holds out `back`, `buffer_overflow`, `portsweep`, and `warezclient`; the threshold-calibration subset holds out `guess_passwd`, `rootkit`, `satan`, and `smurf`. The attack-family names are only used to construct the split. They are not model inputs: all classifiers are trained for binary normal-versus-attack classification.

Normal connections are also assigned to the validation subsets, without reusing the same normal rows across the two subsets. The official test file is loaded for final evaluation and is not used by the code to fit model parameters or select a threshold.

The primary threshold policy selects the threshold that achieves the highest recall while keeping the **threshold-calibration false-positive rate at or below 3%**. This is a target on the calibration subset, not a guarantee that the test false-positive rate will also be 3% or lower. Different attack distributions can produce different test results.

The project has been developed through repeated evaluation runs. For that reason, the reported `KDDTest+` metrics should be described as the observed results for this project, rather than claiming that the official test set was an untouched benchmark throughout the entire development history.

## Fisher Score feature selection

Fisher Score ranks attributes according to how well their values separate the normal and attack classes relative to the variation within each class. A higher score indicates that the attribute provides stronger one-dimensional class separation in the fitting data.

For numeric attributes, the score is calculated from the class means and within-class variation. For the three categorical attributes, each attribute is one-hot encoded for the purpose of ranking; the attribute receives the maximum Fisher Score among its category-indicator columns. This keeps the feature ranking at the level of the **original 41 attributes**, not individual encoded columns.

The ranking is calculated using fitting data only. The 17 attributes with the highest scores are selected, and the same set is used for both the Fisher + ANFIS and Fisher + Decision Tree experiments. The selected features are written to `results/top17_features.csv`.

Feature selection is included as an experiment because a smaller input set can reduce the amount of information the models process and can make some systems easier to manage. However, reducing the number of features does not guarantee better results. It may remove information needed to detect certain attacks, so both the full and reduced feature paths are evaluated.

## The four experiments

The experiment design separates the effect of the classifier from the effect of feature selection.

### Experiment 1 — ANFIS with all 41 attributes

This is the full-feature ANFIS baseline. It uses the 41 original traffic attributes, which become 116 encoded model inputs after preprocessing. It uses the feature-weighted, first-order Sugeno ANFIS described below.

This experiment measures how the ANFIS model performs when it has access to the complete original attribute set.

### Experiment 2 — Fisher Score + ANFIS with 17 attributes

Fisher Score selects the top 17 original attributes from the fitting subset. The selected attributes become 92 encoded model inputs after preprocessing. ANFIS uses the same modelling approach as Experiment 1, but works with the reduced input set.

Comparing Experiments 1 and 2 shows whether feature reduction changes attack detection, false alarms, and overall performance for ANFIS.

### Experiment 3 — Decision Tree with all 41 attributes

This is the conventional machine-learning baseline. A scikit-learn `DecisionTreeClassifier` uses all 41 original attributes, represented by 116 encoded columns. The code compares settings for the split criterion, maximum tree depth, and minimum number of samples per leaf using the model-selection subset.

Comparing Experiment 1 with Experiment 3 gives a comparison between the full-feature ANFIS and the full-feature Decision Tree.

### Experiment 4 — Fisher Score + Decision Tree with 17 attributes

The Decision Tree uses the same 17 Fisher-ranked original attributes as Experiment 2. This makes it possible to compare the effect of feature reduction on the conventional baseline as well as on ANFIS.

Comparing Experiment 2 with Experiment 4 gives a classifier comparison using the same reduced feature set.

## How the ANFIS model works

ANFIS combines fuzzy rules with trainable parameters. In this project, it is a **feature-weighted, first-order Sugeno ANFIS** designed to produce a score for binary classification.

### 1. Fuzzy-rule initialisation

The initial rule centres are created using `MiniBatchKMeans` clustering. Normal and attack examples in the fitting data are clustered separately, and the resulting centres are combined to initialise the rule base. Only fitting labels are used for this initialisation.

The initial Gaussian membership-function widths are estimated from the variation within the clusters. This gives the fuzzy system an initial description of different regions of the encoded input space.

### 2. Gaussian membership functions

Each rule has Gaussian membership functions across the input dimensions. These functions represent how close a connection is to the region represented by a fuzzy rule. The rule firing strengths depend on the normalised distances from the rule centres, the membership widths, learned feature-relevance weights, and a trainable temperature parameter.

### 3. Feature relevance

The ANFIS model learns a relevance weight for each encoded input dimension. These weights alter how strongly each dimension contributes to the distance used for rule activation. They are learned during fitting and are not based on the official test labels.

This mechanism is intended to let the fuzzy rules place more emphasis on useful inputs and less emphasis on less informative ones. The learned weights are part of this project's ANFIS design; feature weighting is not a requirement of every classical ANFIS formulation.

### 4. Sugeno consequents and hybrid learning

Each rule has a first-order linear Sugeno consequent. The output of the full ANFIS is a weighted combination of these rule outputs, with the normalised rule firing strengths determining their contributions.

Training alternates between two parameter-update steps:

- **Regularised least squares:** estimates the linear consequent coefficients for the fuzzy rules.
- **Gradient-based learning:** updates the fuzzy-rule centres, Gaussian widths, temperature, and feature-relevance parameters.

The regularisation value used by the least-squares step is selected during model selection. Gradient updates use Adam, and early stopping monitors validation performance to retain a suitable model state.

The ANFIS output is a continuous **Sugeno score**. It is not assumed to be a calibrated probability. The selected threshold converts the score into a class prediction: scores at or above the threshold are classified as attacks, and lower scores are classified as normal.

### 5. ANFIS model selection

For each feature path, the code evaluates 28 combinations of rule count and ridge regularisation. The rule-count candidates are 4, 6, 8, 12, 16, 24, and 32; the ridge values are 0.00001, 0.0001, 0.001, and 0.01.

To keep this search manageable, candidate configurations are compared using a stratified sample of up to 10,000 fitting rows. The selected configuration is then trained on all 106,463 fitting rows in the reported run. The model-selection validation data are used to monitor performance and select the model state. Threshold calibration is performed separately.

## How the Decision Tree works

The Decision Tree learns a series of feature-based splits. Each split divides the data into smaller groups, and the final leaf assigns a class distribution used to calculate the predicted attack probability.

The code compares 18 configurations for each feature path:

- Split criterion: `gini` or `entropy`
- Maximum depth: unlimited, 20, or 40
- Minimum samples per leaf: 1, 2, or 5

The chosen tree is the one with the best model-selection results under the project's validation policy, with F1-score, recall, and false-positive rate used to rank candidates. The final classification threshold is chosen using the separate threshold-calibration subset.

The Decision Tree is useful as a baseline because it is familiar, relatively fast to train, and produces rule-like decision paths. It does not use fuzzy membership functions or the consequent-learning process used by ANFIS.

## Evaluation metrics

All four experiments are evaluated on the same official `KDDTest+` rows. In the confusion-matrix counts below, **attack is the positive class**.

| Metric | Meaning | Formula |
|---|---|---|
| Accuracy | Fraction of all connections classified correctly | `(TP + TN) / (TP + TN + FP + FN)` |
| Precision | Fraction of predicted attacks that are actual attacks | `TP / (TP + FP)` |
| Recall / true positive rate (TPR) | Fraction of all attacks detected | `TP / (TP + FN)` |
| Attack F1-score | Harmonic mean of attack precision and recall | `2 × Precision × Recall / (Precision + Recall)` |
| False-positive rate (FPR) | Fraction of normal connections incorrectly flagged as attacks | `FP / (FP + TN)` |
| Training time | Time for the final model fit | Measured in seconds |
| Tuning time | Time spent comparing model configurations | Measured in seconds |

The confusion-matrix terms are:

- **True positive (TP):** an attack correctly identified as an attack.
- **True negative (TN):** a normal connection correctly identified as normal.
- **False positive (FP):** a normal connection incorrectly flagged as an attack.
- **False negative (FN):** an attack incorrectly classified as normal.

No single metric describes the entire system. High precision means that alerts are usually real attacks, while high recall means that a larger share of attacks is found. A low false-positive rate is useful because it limits unnecessary alerts. F1-score summarises the balance between precision and recall.

## Results

The following table reports the primary operating point from the supplied `four_experiment_results.csv`. Percentages are rounded to two decimal places; the confusion-matrix counts are the recorded integer counts.

| Experiment | Original attributes | Encoded columns | Accuracy | Precision | Attack recall | Attack F1-score | FPR |
|---|---:|---:|---:|---:|---:|---:|---:|
| 1. ANFIS, all features | 41 | 116 | **80.00%** | 91.12% | **71.87%** | **80.36%** | 9.26% |
| 2. Fisher + ANFIS, top 17 | 17 | 92 | 79.08% | 95.69% | 66.24% | 78.28% | 3.94% |
| 3. Decision Tree, all features | 41 | 116 | 72.45% | 96.67% | 53.45% | 68.84% | 2.43% |
| 4. Fisher + Decision Tree, top 17 | 17 | 92 | 72.47% | **97.96%** | 52.73% | 68.56% | **1.45%** |

### Confusion-matrix counts

| Experiment | TN | FP | FN | TP |
|---|---:|---:|---:|---:|
| 1. ANFIS, all features | 8,812 | 899 | 3,610 | 9,223 |
| 2. Fisher + ANFIS, top 17 | 9,328 | 383 | 4,333 | 8,500 |
| 3. Decision Tree, all features | 9,475 | 236 | 5,974 | 6,859 |
| 4. Fisher + Decision Tree, top 17 | 9,570 | 141 | 6,066 | 6,767 |

### Selected model settings and run times

The settings below were selected by the search process for the reported run. Run times are included for reference and will vary with hardware and the software environment.

| Experiment | Selected settings | Final fit time | Tuning time |
|---|---|---:|---:|
| 1. ANFIS, all features | 12 fuzzy rules; ridge value `0.00001`; best epoch 10 | 14.10 s | 58.59 s |
| 2. Fisher + ANFIS, top 17 | 16 fuzzy rules; ridge value `0.001`; best epoch 22 | 22.09 s | 59.85 s |
| 3. Decision Tree, all features | Gini criterion; no maximum depth limit; minimum 2 samples per leaf | 0.91 s | 13.93 s |
| 4. Fisher + Decision Tree, top 17 | Gini criterion; maximum depth 20; minimum 2 samples per leaf | 0.97 s | 13.11 s |

### What the results show

**Experiment 1: ANFIS using all 41 attributes** achieved the highest accuracy, attack recall, and attack F1-score among the four primary results. It detected 9,223 attacks. Its recall of 71.87% means that it identified roughly 72 out of every 100 attack connections in the official test set. Its false-positive rate was 9.26%, however, and it incorrectly flagged 899 normal connections. This configuration favours attack detection over minimising false alarms.

**Experiment 2: Fisher + ANFIS** had higher precision and a lower false-positive rate than the full-feature ANFIS. It correctly flagged fewer normal connections as attacks: 383 false positives compared with 899 in Experiment 1. It also detected fewer attacks, with 8,500 true positives instead of 9,223, and its recall fell from 71.87% to 66.24%. This shows the trade-off created by reducing the original input set: fewer false alerts, but also more missed attacks. Among the two ANFIS configurations, this is the lower-false-alarm option in the reported comparison.

**Experiment 3: Decision Tree using all 41 attributes** reached 96.67% precision and a 2.43% false-positive rate, but detected only 53.45% of attacks. Its 5,974 false negatives are attacks that were present in the test set but classified as normal. Therefore, the high precision should not be interpreted as meaning that the tree catches most attacks; it means that the alerts it raises are usually correct.

**Experiment 4: Fisher + Decision Tree** produced the highest precision and the lowest false-positive rate of the four primary results. It recorded 141 false positives, but its attack recall was 52.73%, the lowest of the four experiments. It missed 6,066 attacks. Feature selection slightly increased accuracy relative to the full-feature tree, but it did not improve attack recall or F1-score in this run.

### Overall comparison

The results show a consistent difference in operating behaviour. The ANFIS experiments identify a larger fraction of the attacks and achieve higher attack F1-scores. The Decision Tree experiments make fewer false-positive predictions and have higher precision, but miss more attacks.

The full-feature ANFIS has the strongest overall attack-detection result in this table. Fisher + ANFIS is a useful alternative when reducing false alarms matters more, although its measured FPR of 3.94% is still higher than the full-feature Decision Tree's 2.43%. The Fisher-selected Decision Tree has the lowest measured FPR, but also the lowest recall.

Feature selection does not help every classifier in the same way. It reduced the encoded input count from 116 to 92. For ANFIS, this lowered the false-positive rate but also lowered recall and F1-score. For the Decision Tree, it made precision slightly higher and FPR lower, while attack recall and F1-score were slightly lower. These observations apply to the reported run and dataset split; they should not be treated as universal rules for other datasets.

## Threshold selection and the operating-point table

The primary threshold policy is to maximise recall while keeping the false-positive rate at or below 3% on the threshold-calibration subset. The selected threshold is then applied to the continuous ANFIS score or the Decision Tree's positive-class probability when evaluating `KDDTest+`.

The file `operating_point_analysis.csv` provides additional threshold choices:

- **F1-max:** a threshold that maximises attack F1-score on the separate threshold-calibration data.
- **FPR at most 1%:** a threshold selected to maximise recall while meeting a 1% calibration FPR cap.
- **FPR at most 2%:** the same selection rule with a 2% cap.
- **FPR at most 3%:** the primary policy.
- **FPR at most 5%:** the same selection rule with a 5% cap.

These alternatives help show how the alert threshold changes the balance between recall and false alarms. A stricter FPR target usually requires a higher threshold, which tends to reduce false positives but can also reduce recall. The actual test FPR can differ from the calibration FPR because the validation and test samples have different distributions of attack types and traffic.

The main comparison table uses the primary threshold recorded for each experiment in `four_experiment_results.csv`. When preparing final report tables, use outputs generated together from the same run and check that the selected thresholds and confusion-matrix counts are consistent between the two CSV files.

## Output files

After a successful run, the `results/` directory contains the following three CSV files and three plots.

### CSV files

**`four_experiment_results.csv`**

The main results table. It records the feature counts, encoded-column counts, classifier, selected threshold, tuning/training times, selected ANFIS or tree settings, validation summaries, test metrics, and confusion-matrix counts for all four experiments. This is the principal file to use when preparing the results section of the report.

**`operating_point_analysis.csv`**

A threshold sensitivity table. It records validation and test metrics for the F1-max threshold and for thresholds selected under the 1%, 2%, 3%, and 5% validation-FPR policies. Use it to explain how threshold selection changes the recall/false-alarm balance.

**`top17_features.csv`**

The 17 original attributes selected by Fisher Score using fitting data. These are original NSL-KDD attributes, not the one-hot encoded columns.

### Plot files

**`metric_comparison.png`** compares accuracy, attack F1-score, and attack recall for the four experiments. The values printed above each bar make the differences easier to read.

**`fpr_comparison.png`** compares false-positive rates. The vertical axis is scaled to make lower-rate differences visible, and each bar is labelled with its measured value.

**`confusion_matrices.png`** displays the four confusion matrices. These show the numbers of correctly and incorrectly classified normal and attack connections for each model.

The script prints the classification report for each experiment in the terminal as well. Other intermediate tuning records are kept in memory and are not exported as additional CSV files.

## Reproducibility and implementation notes

- The random seed is set to `42` for NumPy, PyTorch, clustering, and the relevant data split operations.
- The ANFIS candidate search uses up to 10,000 fitting rows per feature path; the selected configuration is subsequently fitted using all available fitting rows.
- All four experiments use the same fitting/validation/test partitions and the corresponding full or selected feature path.
- Fisher ranking is calculated from fitting data only.
- Preprocessing parameters are fitted on fitting rows only.
- The `difficulty` column is not included as a feature.
- ANFIS outputs are continuous Sugeno scores, not calibrated probabilities. The output threshold is selected on validation data.
- Training and tuning times depend on the CPU/GPU, memory, and software environment. They may differ between computers.

## Limitations

1. **Binary classification:** the target distinguishes normal traffic from all attack traffic. It does not identify a specific attack type in the final prediction.
2. **Dataset scope:** NSL-KDD is a labelled benchmark dataset. Its results do not by themselves establish performance on live network traffic or current attack patterns.
3. **Different attack distributions:** withholding attack families during validation makes generalisation to unfamiliar attack types part of model selection. Even so, performance can change when the distribution of attacks and normal traffic changes.
4. **False-positive-rate target:** the 3% cap is applied to threshold-calibration data. The test results show that a validation cap is not a guarantee of the same FPR on unseen data.
5. **Threshold sensitivity:** the reported metrics depend on the chosen threshold. The operating-point table should be used to understand the effect of different alert policies.
6. **Repeated project development:** `KDDTest+` is not used by the code to fit parameters or choose thresholds within an individual run. Since test results have been reviewed during project development, describe the displayed figures as results observed during development, not as a benchmark that was consulted only once.

## Methodological reference

The project uses the following sources for the dataset and broad methodological context:

- **NSL-KDD dataset:** [HoaNP/NSL-KDD-DataSet](https://github.com/HoaNP/NSL-KDD-DataSet)
- **ANFIS and feature-selection design reference:** [Springer article](https://link.springer.com/article/10.1007/s44163-026-01217-z)

The code implements its own feature ranking, attack-family validation design, hybrid-learning ANFIS, threshold calibration, and four-experiment comparison. The results reported here are generated by this implementation and should not be presented as a direct reproduction of the reference paper's metrics.

## Summary

This project compares a feature-weighted Sugeno ANFIS and a Decision Tree for binary intrusion detection on NSL-KDD. It evaluates each classifier using all 41 original attributes and a reduced set of 17 Fisher-ranked attributes. In the supplied test results, full-feature ANFIS achieved the highest attack recall and F1-score, while Fisher + Decision Tree achieved the lowest false-positive rate and the highest precision. The results demonstrate why intrusion-detection systems should be compared using recall, precision, F1-score, and false-positive rate together, rather than accuracy alone.
