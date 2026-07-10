# PM-CAST-D

`PM_CAST_D` is the Prime-Modular CP-Anchored Spectral Transport Distribution
model. It is a practitioner model whose forecast remains a positive weighted
average transport anchored to `CP_REPO_FRESH`.

For a lag path `x in R_+^22`, define

```text
z = log(x + eps) - mean(log(x + eps)) 1,
M_C = I - C' (C C')^+ C,
H_tau = Re(Psi exp(-tau Lambda) Psi* M_C).
```

The PM energy is normalized by one scalar per path, preserving relative heat
weights:

```text
e = H_tau z / sqrt(mean((H_tau z)^2)),
beta_l proportional to exp(-kappa(l-1)),
w_l = beta_l exp(gamma e_l) / sum_j beta_j exp(gamma e_j).
```

The transported and reference averages are

```text
A_PM = sum_l w_l x_l,
A_0  = sum_l beta_l x_l,
d_PM = log((A_PM + eps)/(A_0 + eps)).
```

The CP-relative point action is

```text
m = m_CP exp(g * clip(b + alpha d_PM, -c, c)),
```

where `g` is an optional high-CP tail-protection factor computed from the
strictly prior expanding CP rank. When `b=alpha=0`, the prediction is exactly
`CP_REPO_FRESH`.

## Frozen Selection

- `n=22`, `warmup=600`, `seed=42`.
- Only origins before the first warmup-600 OOS origin may select parameters.
- Four chronological pre-OOS folds are used.
- The selection score maximizes the worst of SMAPE, MAE, and MSE normalized
  improvements after a fold-dispersion penalty.
- The final OOS period is evaluated once after the configuration is frozen.
- RMSE is not separately tuned because it has the same ordering as MSE on an
  identical row set.

## Matched Controls

`SPECTRAL_RANDOM_CAST_D` and `CONTIGUOUS_CAST_D` preserve the PM operator's
singular spectrum, rank, CP-null input space, action grid, fold boundaries, and
tuning budget. `CAST_D_CALIBRATION_ONLY` removes the transport ratio and
isolates generic CP-relative calibration.

The implementation and workflow live under
`outputs/pm_cast_d_warmup600/` and `.github/workflows/pm_cast_d_warmup600.yml`.

