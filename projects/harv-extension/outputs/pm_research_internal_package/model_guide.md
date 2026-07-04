# Model Guide

## CP_REPO_FRESH
The corrected benchmark is `CP_REPO_FRESH = RV/HAR + contig_prime_modulo(vol.copy(), n=22, per_day_normalize=False)`. It uses repo `RV*` and `CP_*` feature columns and is the denominator for the corrected repo-CP ladder.

## Raw PM and Raw CP+PM
`RAW_PM` tests standalone prime-modulo residue averages from `add_prime_modulo_terms(..., n=22)`. `RAW_CP_REPO_PLUS_PM` adds raw PM to the repo CP controls and replicates the older incremental test pattern: noisy overall, useful in some entry/recent-spike regimes.

## OPS-R
OPS-R uses centered prime-residue contrasts as an incremental shape layer beyond repo CP. The ridge version is the controlled diagnostic; unregularized residue models are noisy.

## LRPM
LRPM localizes residue contrasts inside lag neighborhoods. The current unregularized `CP_REPO_LRPM` is rank deficient and diagnostic only. `CP_REPO_RIDGE_LRPM` is the paper-eligible repair path for future runs.

## Recent Slope, Haar, OPS-C
These are generic path-shape controls. If they match or beat PM-specific residue models, the evidence is path-shape rather than prime-specific.

## Gated OPS-C and OPS-HG
`CP_REPO_GATED_RIDGE_OPS_C` is the practical cleaned shape model. `CP_REPO_OPS_HG` adds heavily shrunk OPS-R to gated OPS-C. If HG loses to gated OPS-C, the PM-residue addition should be rejected.

## Placebos
Random residues, shuffled PM, and random gates test whether prime residue structure, true lag order, and the real gate matter.

## Perturbation Identity
For challenger perturbation `delta_t = yhat_model_t - yhat_CP_t` and CP residual `e_t = y_t - yhat_CP_t`, squared-loss improvement is `2 e_t delta_t - delta_t^2`.

Interpretation: a PM/shape model helps only when its correction aligns with the CP residual enough to overcome its perturbation penalty.
