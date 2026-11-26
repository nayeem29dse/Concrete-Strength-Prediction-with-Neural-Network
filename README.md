# Concrete Compressive Strength Prediction (PyTorch)

Predicting concrete compressive strength (MPa) from mix composition and age, starting from a linear model written by hand and ending with a cosine-scheduled neural network ensemble, evaluated with 5-fold cross-validation.

**Final result (5-fold CV): RMSE ≈ 4.3 MPa, R² ≈ 0.93**, versus RMSE 10.9 MPa and R² 0.54 for the linear baseline.

---

## 1. Data

- UCI Concrete Compressive Strength dataset: 1030 rows, no missing values.
- 8 inputs: cement, blast furnace slag, fly ash, water, superplasticizer, coarse aggregate, fine aggregate (kg per m³) and age (days).
- Target: compressive strength in MPa.
- Split: 80/20 train/test (824/206 rows), shuffled with a fixed seed. `StandardScaler` is fit on the training rows only, so no test information leaks into scaling.

## 2. The journey

### Step 1: Linear regression from scratch (baseline)
Implemented `y = XW + b`, MSE loss and gradient descent manually with autograd.

| | RMSE | R² |
|---|---|---|
| Train | 10.29 | 0.626 |
| Test | 10.93 | 0.541 |

**Finding:** train and test are close, so this is **underfitting**, not overfitting. Strength depends nonlinearly on the inputs (age saturates, water/cement interactions), and a straight line can't capture that.

### Step 2: Neural network (8 → 64 → 32 → 1, ReLU, Adam)
Test RMSE dropped to **5.35 MPa** (R² 0.890) after 1000 epochs, with train RMSE 3.69. Test loss bottomed out around epoch 650 and then crept up, which is mild overfitting.

**Mistake caught:** my first "1000-epoch" run actually continued training an already-trained network (the model and optimizer cell wasn't re-run), so the epoch-0 loss was 26 instead of about 1500. From then on, model, optimizer and loop live in one cell, so every run starts fresh.

### Step 3: Weight decay, and why single-run comparisons mislead
Single-seed runs suggested weight decay helped (test RMSE 5.35 → 4.63 at 1e-2). A proper search with a validation split and 3 seeds said otherwise:

| weight decay | Val RMSE (3 seeds) |
|---|---|
| 0 | 6.08 ± 0.26 |
| 1e-3 | 6.17 ± 0.02 |
| 1e-2 | 6.38 ± 0.45 |
| 1e-1 | 6.26 ± 0.08 |

All four are tied within the seed noise, and the 3-seed test RMSE with no decay was **5.14 ± 0.22**. The earlier "gain" was mostly seed luck plus choosing settings by watching the test set.

**Lesson:** a single seed on a 206-row test set can move RMSE by about 0.2–0.5 MPa. Differences smaller than that are not evidence.

### Step 4: Feature engineering and a deeper network
Added `water/cement ratio` and `log(age)` (10 features total) and moved to an 80 → 40 → 20 → 1 network. Test RMSE improved to about **4.4 MPa**.

- Validation search (Adam, 3 seeds): weight decay now showed a clear trend, 0 → 7.33, 1e-3 → 6.63, 1e-2 → 6.17, **1e-1 → 5.60**. The bigger network has far more capacity and benefits from the penalty, unlike the small one.
- Test RMSE with wd = 0.1: **4.38 ± 0.22**.
- **Caveat:** I changed features and network size at the same time. From my own runs the improvement came mostly from the bigger network, with the engineered features adding little, but I did not save a controlled 8-vs-10 feature comparison, so treat that as an observation, not a measured result.

### Step 5: Things that did not help
- **AdamW** (decoupled decay), grid 0.1–3.0: best val RMSE 6.14 and test RMSE 4.97 ± 0.42, worse than plain Adam with L2 decay (val 5.60, test 4.38). Dropped.
- **Stronger Adam decay** (0.3): test RMSE 4.47 on one seed, tied with 0.1. Train error rose and the gap shrank, but test did not improve. A smaller train/test gap is a diagnostic, not the goal.
- **Mini-batch training** (batch 64): CV ensemble RMSE 4.58, worse than full-batch (4.37), losing in 3 of 5 folds.

### Step 6: Ensembling (the biggest late gain)
Single networks varied from 4.12 to 4.66 MPa test RMSE depending only on the random seed. Averaging the predictions of several nets reduces that variance.

| Model | Test RMSE |
|---|---|
| Single nets (5 seeds) | 4.357 ± 0.204 |
| 5-net ensemble | **4.067** |

### Step 7: Honest evaluation with 5-fold cross-validation
Because I had tuned the design while looking at the same test set, I re-estimated performance with 5-fold CV (scaler fit inside each fold, 3-net ensemble per fold):

| Model | RMSE | R² |
|---|---|---|
| Single net | 4.64 ± 0.35 | n/a |
| 3-net ensemble | **4.37 ± 0.37** | **0.931 ± 0.011** |

The ensemble beat the single net in all 5 folds (gain 0.17–0.32 MPa), so the improvement is consistent, not a lucky split.

### Step 8: Learning-rate schedule
Controlled comparison on identical folds and seeds:

| Variant | Ensemble RMSE | Folds won | Mean gain |
|---|---|---|---|
| Baseline | 4.374 ± 0.374 | n/a | n/a |
| Cosine annealing | 4.312 ± 0.362 | 4/5 | +0.062 |
| Mini-batch | 4.581 ± 0.238 | 2/5 | −0.207 |

Cosine is a small, consistent, free improvement (it also removes the late-epoch loss spikes seen with a constant learning rate of 0.01), but the gain is far inside the fold-to-fold spread. I kept it because it costs nothing, not because it is proven.

## 3. Final configuration and why

| Choice | Value | Why |
|---|---|---|
| Features | 8 original + water/cement + log(age) | Cheap and harmless, though the measured benefit is unproven |
| Architecture | 10 → 80 → 40 → 20 → 1, ReLU | Bigger than the first net; the extra capacity was where the real improvement came from |
| Optimizer | Adam, lr 0.01 | Converged reliably |
| Regularization | L2 weight decay 0.1 | Clear validation gain for the larger net; AdamW was worse |
| Schedule | Cosine annealing over 1500 epochs | Small, consistent gain; removes spikes |
| Training | Full batch | Mini-batch was worse |
| Final model | Ensemble of 5 nets (different seeds) | Largest late gain; cuts seed variance |

The deployment model is retrained on all 1030 rows. Its expected accuracy is the CV estimate, since no unseen data remains.

## 4. Limitations

- Only 1030 rows with measurement noise; the error floor is probably near 3.5–4 MPa, and further tuning would be chasing noise.
- Architecture and weight decay were chosen while looking at the same rows later used for CV, so the CV figure is slightly optimistic.
- CV folds share training data, so the ± values are a rough spread, not a strict confidence interval.
- The CV experiments used 3-net ensembles. The final 5-net cosine configuration was evaluated on the held-out test split, not re-run through CV, so expect it to be similar to or marginally better than the 4.31 figure.
- The engineered-feature contribution was not isolated in a saved ablation.

## 5. What next can be done

- Try gradient-boosted trees (XGBoost/LightGBM) as a strong tabular baseline.
- Collect or merge more data, since variance and noise, not model design, limit accuracy now.
