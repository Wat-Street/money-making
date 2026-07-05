# PM_QDK_2

Loss-native Prime-Modular Quotient Diffusion Kernel forecaster.

This exploratory challenger uses log-centered volatility paths as measures on the prime torus, applies multiscale heat diffusion, removes CP-explainable PM geometry with train-only quotient residualization, and forecasts with a kernel in `(level, CP state, quotient PM geometry)`.

The point forecast is a weighted Bayes action for SMAPE, not an ordinary least-squares feature add-on.
