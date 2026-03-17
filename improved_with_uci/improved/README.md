# SmartMonitoringSystem 

Temporal Disease Progression Modelling with Behavioural Robustness Evaluation

---

## Overview

This codebase implements the framework described in:

> **Evaluating Behavioural Robustness in Temporal Health Risk Prediction under Simulated Lifestyle Intervention**

The system extends static disease prediction to **temporal trajectory modelling**, evaluating how LSTM and Transformer architectures respond to simulated lifestyle interventions.

---



### Critical Bug Fixes

| Bug | Original | Fixed |
|-----|----------|-------|
| **AUC inconsistency** (Tables 6 vs 10 in paper) | Ablation used multi-output AUC → inflated to 0.92–0.98 while main eval showed 0.57–0.61 | All evaluations now use `y[:,0]` (7-day risk) consistently |
| **Intervention had no effect** | Post-hoc `+0.5` constant offset bypassed progression model | Two parallel trajectory branches with exponential lag ramp-up (physiological inertia) |
| **Uncertainty estimation broken** | `model.train()` barely activated dropout | `enable_mc_dropout()` explicitly sets all `nn.Dropout` to train mode |
| **Label leakage risk** | `i % 5` deterministic resilience tied to row index | `Beta(2,5)` sampled independently per patient |
| **Ablation model never trained in main** | `main.py` only called 2 of 3 training scripts | All 3 models trained before evaluation |
| **Pandas deprecation warnings** | `fillna(inplace=True)` | Fixed for pandas 2.0+ CoW semantics |

### Features

- **NHANES-calibrated dataset** (`utils/nhanes_synthetic.py`): N=2000, 8 clinical features (Glucose, HbA1c, SBP, DBP, BMI, Age, HDL, Triglycerides), ADA 2021 diagnostic criteria, 11.9% prevalence matching NHANES 2017–18
- **Cross-dataset validation** (`analysis/cross_dataset_validation.py`): Compares Pima vs NHANES results for generalisability
- **Bootstrap 95% CI** on all AUC values
- **Wilcoxon signed-rank test** for intervention significance
- **Expected Calibration Error (ECE)**
- **Proper MC-Dropout** with `enable_mc_dropout()`
- **KS test** for synthetic distribution validation
- **Calibration curves** (reliability diagrams)
- **Intervention SHAP delta** (Δφ per feature)
- **Publication-quality figures** with error bars and uncertainty bands

---

## Project Structure

```
SmartMonitoringSystem/
│
├── main.py                          # Master pipeline (runs all 4 steps in order)
│
├── data/
│   ├── raw/
│   │   ├── diabetes.csv             # Pima Indians dataset (original)
│   │   └── nhanes_synthetic.csv     # NHANES-calibrated dataset (generated)
│   └── processed/
│
├── models/
│   ├── lstm_model.py                # LSTM + MC-Dropout + user embeddings
│   └── transformer_model.py        # Transformer + MC-Dropout + optional personalisation
│
├── utils/
│   ├── preprocessing.py             # Pima loader with stratified train/val/test split
│   ├── clinical_temporal.py         # Pima → longitudinal trajectories (baseline + intervention)
│   ├── nhanes_synthetic.py          # NHANES-calibrated dataset generator
│   ├── nhanes_temporal.py           # NHANES → longitudinal trajectories
│   ├── metrics.py                   # AUC-CI, ECE, BR, Wilcoxon, velocity, uncertainty
│   ├── dataset_builder.py           # Simple synthetic builder (unit testing only)
│   └── temporalize_real.py          # Noisy temporalization for real datasets
│
├── training/
│   ├── train_lstm.py                # Train LSTM on Pima
│   ├── train_transformer.py         # Train personalised Transformer on Pima
│   ├── train_transformer_no_personal.py  # Ablation: no identity embeddings
│   └── train_nhanes.py              # Train all models on NHANES dataset
│
├── evaluation/
│   ├── compare_models.py            # Master evaluation script (all paper results)
│   ├── ablation.py                  # Ablation study (consistent AUC metric)
│   ├── counterfactual.py            # Intervention sensitivity (lagged trajectories)
│   ├── uncertainty.py               # MC-Dropout uncertainty estimation
│   ├── risk_trajectory.py           # Baseline vs intervention trajectory plots
│   ├── risk_velocity.py             # Risk velocity + temporal risk series
│   ├── shap_explain.py              # Temporal SHAP + intervention Δφ
│   ├── onset.py                     # Time-to-onset estimation
│   ├── figures.py                   # Publication-quality figure generation
│   ├── temporal_plot.py             # Feature evolution plots
│   └── cv_plot.py                   # Cardiovascular risk progression plot
│
└── analysis/
    ├── realism_validation.py        # Descriptive stats + correlation + normality
    ├── validate_synthetic.py        # KS test: synthetic vs original distributions
    └── cross_dataset_validation.py  # Pima vs NHANES comparison (generalisability)
```

---

## Quickstart

### 1. Install dependencies

```bash
pip install torch numpy pandas scikit-learn scipy matplotlib shap
```

### 2. Run the full pipeline (Pima dataset)

```bash
python main.py
```

This will:
1. Train LSTM on Pima
2. Train Transformer (personalised) on Pima
3. Train Transformer (no personalisation) on Pima
4. Run full evaluation → saves all figures to `figures/`

### 3. Run NHANES cross-dataset validation

```bash
python training/train_nhanes.py
python analysis/cross_dataset_validation.py
```

### 4. Run realism validation only

```bash
python analysis/realism_validation.py
python analysis/validate_synthetic.py
```

---

## Datasets

### Pima Indians Diabetes (original)
- N = 768, 8 features → 6 temporal features after engineering
- Single-sex (female), Pima Indian heritage cohort
- Source: Smith et al., 1988 / UCI ML Repository

### NHANES-calibrated Synthetic (new)
- N = 2000, 8 features: Glucose, HbA1c, SBP, DBP, BMI, Age, HDL, Triglycerides
- Mixed-sex, multi-ethnic representation via published NHANES distributions
- Outcome: ADA 2021 criteria (FPG ≥ 126 mg/dL OR HbA1c ≥ 6.5%)
- Prevalence: ~11.9% (matches NHANES 2017–18 reported diabetes prevalence)
- Correlation structure from published NHANES inter-variable correlations

**Note**: The NHANES-calibrated dataset is a synthetic approximation whose
marginal distributions and pairwise correlations match published NHANES
summary statistics. It is NOT the real NHANES microdata (which requires a
CDC data use agreement). It is suitable for methodology validation and
reproducible benchmarking.

---

## Key Metrics

| Metric | Description | Location |
|--------|-------------|----------|
| AUC + 95% CI | Bootstrap confidence interval on ROC-AUC | `utils/metrics.py` |
| ECE | Expected Calibration Error | `utils/metrics.py` |
| BR | Behavioural Robustness: mean \|R'_t - R_t\| | `utils/metrics.py` |
| BR_velocity | Velocity-based robustness metric | `utils/metrics.py` |
| Wilcoxon p | Statistical significance of intervention effect | `utils/metrics.py` |
| Uncertainty σ | MC-Dropout prediction variance | `evaluation/uncertainty.py` |
| CoV | Coefficient of Variation (σ / μ) | `utils/metrics.py` |
| Δφ | SHAP attribution shift under intervention | `evaluation/shap_explain.py` |

---

## Intervention Model

The intervention is implemented as **two parallel longitudinal trajectories**:

- `X_base`: Baseline trajectory (declining activity trend)
- `X_interv`: Intervention trajectory (activity boost starting at t=10)

The activity increase ramps up exponentially:

```
act_interv(t) = act_base(t) + Δ × (1 - exp(-(t - t_start) / τ))
```

Where:
- `Δ = 0.30–0.35` (intervention magnitude)
- `τ = 5–6` (lag time constant, physiological inertia)
- `t_start = 10`

This models the clinically observed delay in physiological response to
lifestyle modification (DPP trial: HbA1c reduction emerges over months).

---

## Citation

If you use this code, please cite:

```
Nanda, M. & Nagpall, H. (2026). Evaluating Behavioural Robustness in 
Temporal Health Risk Prediction under Simulated Lifestyle Intervention.
```

---

## License

MIT License. NHANES-calibrated dataset is synthetic and freely reusable.
Original Pima dataset: UCI ML Repository (public domain).
