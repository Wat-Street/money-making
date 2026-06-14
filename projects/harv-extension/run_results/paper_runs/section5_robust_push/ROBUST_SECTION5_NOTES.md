# Robust Section 5 Run Notes

Source run: https://github.com/Wat-Street/money-making/actions/runs/27172485765

The run completed successfully across all 10 tickers and all 47 model specifications:
470 model-by-ticker jobs completed, with zero failures.

## Main Takeaways

- PM still improves on HAR, but only modestly: +0.765 SMAPE points on average.
- CP is the strongest prime-family model: +1.290 SMAPE points on average versus HAR.
- CP's gains rise from 3 to 9 extra features and then flatten, so most of its benefit appears by roughly 6-9 engineered features.
- PM is flat across feature counts, which weakens the claim that adding more PM buckets is the driver.
- Equal-window and prefix-window controls are very close to CP at matched feature counts. This suggests the main empirical ingredient is local ordered segmentation, not necessarily the exact prime construction.
- Prime modulo does not materially beat composite or non-coprime modulo controls at matched feature counts. This means the CRT/prime argument is mathematically clean, but the empirical evidence does not show that primality itself is the main source of lift.
- Random controls do not explain CP: CP_FC6 and CP_FC9 beat all matched RAND/CRS samples in this sparse seed set. PM is much less distinctive versus random controls.
- EWMA and GARCH(1,1) perform materially worse than HAR in this 5-minute realized-volatility setup.
- HARQ is only mildly better than HAR in paired SMAPE and is noisy.
- HAR-TCJ has much lower raw SMAPE, but it evaluates on a shorter aligned sample. On paired timestamps, it is only slightly better than HAR and worse than CP on average. Do not use raw HAR-TCJ SMAPE as a direct headline comparison.

## Recommended Paper Framing

Main text should emphasize CP, not PM:

> The robust ablation evidence points to ordered local segmentation as the main source of improvement. CP remains competitive against matched feature-count, random, and shifted-window controls, while PM's advantage is smaller and is not clearly separable from non-prime modulo controls.

Recommended Section 5 structure:

1. Parameter efficiency and feature-count curve.
   Show CP's improvement saturating around 6-9 features. Mention PM's flat curve.

2. Contiguous-window controls.
   Explain that contiguous ordered windows are broadly useful; CP is competitive but not uniquely dominant versus equal/prefix windows.

3. Prime versus non-prime modulo.
   Treat this as a negative/qualifying result. Do not claim primality is empirically essential.

4. Random controls.
   Use this as the strongest robustness evidence for CP: random and contiguous-random features do not reliably match CP at 6-9 features.

5. External baselines.
   Keep this in the appendix or a short paragraph. EWMA/GARCH are worse; HARQ is weak; HAR-TCJ must be interpreted with paired comparisons because of sample alignment.

## Files

- `robust_summary/tables/robust_summary_by_model.csv`: raw metric summaries by model.
- `robust_summary/tables/robust_summary_vs_baselines.csv`: paired comparisons versus HAR, PM, and CP.
- `robust_summary/tables/robust_feature_count_curve.csv`: feature-count curve inputs.
- `robust_summary/tables/robust_contiguous_windows_summary.csv`: contiguous-window controls.
- `robust_summary/tables/robust_prime_vs_nonprime_summary.csv`: prime/composite/non-coprime modulo controls.
- `robust_summary/tables/robust_random_controls_summary.csv`: RAND/CRS matched controls.
- `robust_summary/figures/*.pdf`: generated figures; usable as drafts, but should be restyled before paper inclusion.
