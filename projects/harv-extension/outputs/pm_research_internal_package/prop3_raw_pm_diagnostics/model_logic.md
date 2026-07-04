# Prop 3 Model Logic

Correct CP anchor: `CP_REPO_FRESH = RV/HAR + contig_prime_modulo(vol.copy(), n=22, per_day_normalize=False)`.

Correct PM construction: `PM = add_prime_modulo_terms(..., n=22)` using the repo minimal-prime / CRT-style construction. For `n=22`, the relevant minimal primes are expected to include `2, 3, 5`.

Raw CP+PM tests whether adding raw prime-residue averages to the CP/HAR design improves forecasts. It is deliberately not a cleaned OPS model.

The stale `CPB_B*` four-block lag means are diagnostic only and are not the primary CP benchmark.
