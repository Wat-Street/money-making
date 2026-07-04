# Prop 4 Model Logic

Correct CP:

`CP_REPO_FRESH = RV/HAR + contig_prime_modulo(vol.copy(), n=22, per_day_normalize=False)`

Correct PM:

`PM = add_prime_modulo_terms(..., n=22)` using the repo minimal-prime construction.

Main model form:

`y_{t+1} = alpha + beta'C_t + theta'Z_t + eps`

Partial ridge form:

`min ||y - alpha - C beta - Z theta||^2 + lambda_Z ||theta||^2`

Gated/hybrid form:

`y_{t+1} = alpha + beta'C_t + g_t(theta_C'Z_C,t + theta_R'Z_R,t) + eps`

`CPB_B*` four-block lag means are diagnostic only and are forbidden as the primary CP benchmark.
