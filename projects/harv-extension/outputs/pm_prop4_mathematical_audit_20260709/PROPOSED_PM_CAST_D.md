# PM-CAST-D: Prime-Modular CP-Anchored Spectral Transport Distribution

## Purpose

`PM-CAST-D` is an average-based forecasting model. PM is embedded in the averaging operator itself, while `CP_REPO_FRESH` is exactly nested. It is designed to estimate the conditional distribution, then emit one frozen multi-loss point action for the common SMAPE/MAE/MSE/RMSE comparison.

No model can guarantee finite-sample OOS dominance across all four losses. Mean, median, and the SMAPE Bayes action are generally different. The defensible objective is a CP-nested class, train-only selection, and a fresh holdout on which one common point forecast improves all four metrics.

## 1. Exact PM quotient field

For the positive lag path `x in R_+^22`, define

```text
z = log(x + eps) - mean(log(x + eps)) 1,
r = M_C z,
M_C = I - C' (C C')^+ C.
```

Let `U` be an orthonormal real basis obtained from the restricted characters of `Z2 x Z3 x Z5`, and let `mu_k` be the product-cycle Laplacian eigenvalues. The dimensionless PM heat fields are

```text
E_tau(x) = U diag(exp(-tau mu_k)) U' r.
```

Only one scalar train-only normalization is allowed for the entire Hilbert block. Coordinate-wise standardization is forbidden because it removes `exp(-tau mu_k)`.

## 2. PM spectral transport of average weights

Let `beta > 0`, `sum(beta)=1`, be a fixed positive CP/HAR averaging reference. A low-rank state map produces spectral transport scores:

```text
s_l(x) = sum_k theta_k(Cx) [E_tau_k(x)]_l,
w_l(x) = beta_l exp(s_l(x)) / sum_j beta_j exp(s_j(x)).
```

The PM average and its CP reference are

```text
A_PM(x) = sum_l w_l(x) x_l,
A_0(x)  = sum_l beta_l x_l.
```

The CP-nested transported location is

```text
m_T(x) = m_CP(x) [A_PM(x)+eps] / [A_0(x)+eps].
```

If the PM score is zero, the transport temperature is zero, or `A_0=0`, set `m_T=m_CP` exactly.

## 3. Distributional calibration and one common action

Model the CP-relative log outcome

```text
R = log((Y+eps)/(m_CP+eps))
```

with a monotone distributional regression whose location is `log(m_T/m_CP)` and whose scale/tail parameters depend only on low-rank PM heat energy and the CP state. Fit it with expanding, purged, train-only CRPS. The practitioner forecast is restricted to the CP-to-transport segment

```text
a(alpha) = (1-alpha) m_CP + alpha m_T,  0 <= alpha <= alpha_max.
```

Choose one frozen `alpha` rule by minimizing worst normalized validation regret across SMAPE, MAE, and MSE:

```text
min_alpha max_L [R_L(a(alpha))-R_L(m_CP)] / scale_L,
L in {SMAPE, MAE, MSE}.
```

RMSE follows MSE ordering on an identical row set. Entropy/KL regularization on `w` keeps the construction close to the reference average, and `alpha=0` gives exact CP nesting.

## 4. Why this is a useful application exhibit

The forecast remains an average-based system: PM changes the mass assigned to the 22 lag observations through a prime-harmonic transport field. It does not append PM regressors to CP. The distributional layer prevents the SMAPE-only downward shift that caused the current upper-tail MAE/MSE failure.

Application success is not itself proof of prime specificity. `PM-CAST-D` must beat spectral-matched random, shuffled, Haar, composite, fake-torus, and contiguous transport operators under the identical estimator and tuning rule.
