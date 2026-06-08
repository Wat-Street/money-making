import numpy as np
from scipy import stats
import pandas as pd


def build_minimal_primes(n):
    """
    Return the smallest list of distinct primes whose product is at least ``n``.
    """
    if n is None or n <= 1:
        return []

    primes = []
    product = 1
    candidate = 2
    while product < n:
        is_prime = True
        for p in primes:
            if candidate % p == 0:
                is_prime = False
                break
        if is_prime:
            primes.append(candidate)
            product *= candidate
        candidate += 1
    return primes


def restore_index(df, orig_index_name):
    target = orig_index_name if orig_index_name is not None else "index"
    if target not in df.columns:
        raise KeyError(f"Cannot restore index '{target}' — column missing.")
    return df.set_index(target)


def _reset_with_work_index(data):
    orig_index = data.index.name or "index"
    frame = data.reset_index()
    frame["Index"] = np.arange(len(frame))
    return frame, orig_index


def _merge_unique_primes(*prime_lists):
    merged = []
    for prime_list in prime_lists:
        for prime in prime_list:
            if prime not in merged:
                merged.append(prime)
    return merged


def _compute_contiguous_phase_features(frame, value_col, primes, n, per_day_normalize, prefix):
    features = {}
    if not primes:
        return features

    horizon = int(n) if n is not None else 0
    if horizon < 0:
        horizon = 0

    index_series = frame["Index"]

    for k in primes:
        if k <= 0:
            continue
        if "SR" not in frame.columns:
            raise KeyError("Expected column 'SR' (squared return / variance contribution). Add it in calculate_*_realized_volatility.")

        rolling = frame["SR"].rolling(window=k, min_periods=k)
        block_rv = np.sqrt(rolling.sum())

        # per_day_normalize: average RV per block length
        block_metric = (block_rv / k) if per_day_normalize else block_rv
        complete_blocks = max(1, horizon // k)

        base_mask = index_series >= (k - 1)
        for phase in range(k):
            phase_mask = base_mask & (((index_series - phase + 1) % k) == 0)
            block_series = block_metric[phase_mask].dropna()
            if block_series.empty:
                continue
            aggregated = block_series.rolling(window=complete_blocks, min_periods=1).mean()
            col_name = f"{prefix}_k{k}_a{phase}"
            features[col_name] = aggregated

    return features


def _assign_phase_features(frame, feature_map):
    for col_name, series in feature_map.items():
        frame[col_name] = np.nan
        frame.loc[series.index, col_name] = series
        frame[col_name] = frame[col_name].ffill().bfill()


def _compute_prime_modulo_features(value_series, n, primes, prefix, weight_series=None):
    features = {}
    if n is None or n <= 0 or not primes:
        return features

    horizon = int(n)
    lags = list(range(1, horizon + 1))
    lagged_values = pd.concat([value_series.shift(lag) for lag in lags], axis=1)
    lagged_values.columns = lags

    lagged_weights = None
    if weight_series is not None:
        lagged_weights = pd.concat([weight_series.shift(lag) for lag in lags], axis=1)
        lagged_weights.columns = lags

    for prime in primes:
        if prime <= 0:
            continue
        for remainder in range(prime):
            relevant_lags = [lag for lag in lags if lag % prime == remainder]
            if not relevant_lags:
                continue
            subset = lagged_values[relevant_lags]
            if lagged_weights is not None:
                weight_subset = lagged_weights[relevant_lags]
                weight_sum = weight_subset.sum(axis=1)
                numerator = (subset * weight_subset).sum(axis=1)
                feature = numerator / weight_sum
                feature = feature.where(weight_sum > 0)
            else:
                feature = subset.mean(axis=1)
            features[f"{prefix}_k{prime}_r{remainder}"] = feature

    return features


# Strategy 1: Exhaustive Search
def add_exhaustive_terms(data, n):
    for j in range(1, n + 1):
        col_name = f"RV_{j}"
        data[col_name] = data["RV_d"].rolling(window=j).mean()
    return data.replace([np.inf, -np.inf], np.nan)


# Strategy 2: Hamming Codes
def add_hamming_terms(data, n):
    frame, orig_index = _reset_with_work_index(data)
    num_terms = int(np.ceil(np.log2(n))) if n and n > 0 else 0
    for j in range(num_terms):
        col_name = f"RV_bin_{j}"
        frame[col_name] = ((frame["Index"] & (1 << j)) != 0).astype(int) * frame["RV_d"]
    frame = frame.drop(columns=["Index"])
    return restore_index(frame, orig_index)


# Strategy 3: Prime Modulo Classes (core)
def add_prime_modulo_terms(data, n):
    frame, orig_index = _reset_with_work_index(data)
    primes = build_minimal_primes(n)
    feature_map = _compute_prime_modulo_features(
        frame["RV_d"],
        n=n,
        primes=primes,
        prefix="PM",
    )
    for col_name, series in feature_map.items():
        frame[col_name] = series
    frame = frame.drop(columns=["Index"])
    return restore_index(frame, orig_index)


# Strategy 3a: Volume-weighted primes (variant)
def add_volume_weighted_prime_modulo_terms(data, n):
    frame, orig_index = _reset_with_work_index(data)
    horizon = max(int(n), 1)
    volume_mean = frame["Volume"].rolling(window=horizon, min_periods=max(1, horizon // 2)).mean()
    normalized_volume = (frame["Volume"] / volume_mean).replace([np.inf, -np.inf], np.nan).fillna(1.0)
    frame["normalized_volume"] = normalized_volume

    base_market_primes = [2, 5, 23]
    primes = _merge_unique_primes(base_market_primes, build_minimal_primes(n))

    feature_map = _compute_prime_modulo_features(
        frame["RV_d"],
        n=n,
        primes=primes,
        prefix="PMVW",
        weight_series=frame["normalized_volume"],
    )
    for col_name, series in feature_map.items():
        frame[col_name] = series
    frame = frame.drop(columns=["Index"])
    return restore_index(frame, orig_index)


# Strategy 3b: Volume-weighted adaptive primes (variant)
def add_volume_weighted_adaptive_prime_modulo_terms(data, n):
    frame, orig_index = _reset_with_work_index(data)
    horizon = max(int(n), 1)

    volume_mean_short = frame["Volume"].rolling(window=horizon, min_periods=max(1, horizon // 2)).mean()
    frame["normalized_volume"] = (frame["Volume"] / volume_mean_short).replace([np.inf, -np.inf], np.nan).fillna(1.0)

    vol_mean = frame["RV_d"].rolling(window=horizon, min_periods=max(1, horizon // 2)).mean()
    frame["vol_level"] = (frame["RV_d"] / vol_mean).replace([np.inf, -np.inf], np.nan)

    short_std = frame["RV_d"].rolling(window=horizon, min_periods=max(1, horizon // 3)).std()
    long_window = max(horizon * 5, horizon + 1)
    long_std = frame["RV_d"].rolling(window=long_window, min_periods=max(1, long_window // 2)).std()
    frame["vol_of_vol"] = (short_std / long_std).replace([np.inf, -np.inf], np.nan)
    frame["vol_of_vol"] = frame["vol_of_vol"].fillna(1.0).clip(lower=0)

    frame["volume_ratio"] = (frame["Volume"] / volume_mean_short).replace([np.inf, -np.inf], np.nan)

    stress_components = pd.concat(
        [frame["vol_level"], frame["vol_of_vol"], frame["volume_ratio"]],
        axis=1,
    )
    frame["market_stress"] = stress_components.mean(axis=1).clip(lower=0, upper=1).fillna(0.5)

    calm_market_primes = [7, 23]
    stress_market_primes = [2, 3, 5]
    primes = _merge_unique_primes(calm_market_primes, stress_market_primes)

    feature_map = _compute_prime_modulo_features(
        frame["RV_d"],
        n=n,
        primes=primes,
        prefix="PMAdapt",
        weight_series=frame["normalized_volume"],
    )

    high_stress = frame["market_stress"] > 0.7
    low_stress = frame["market_stress"] < 0.3
    medium_stress = ~(high_stress | low_stress)

    prime_activation = {}
    for prime in primes:
        if prime in stress_market_primes:
            if prime == 5:
                prime_activation[prime] = high_stress
            else:
                prime_activation[prime] = high_stress | medium_stress
        else:
            prime_activation[prime] = low_stress

    for col_name, series in feature_map.items():
        segment = col_name.split("_k")[1]
        prime = int(segment.split("_")[0])
        activation = prime_activation.get(prime)
        if activation is not None:
            series = series.where(activation, 0.0)
        frame[col_name] = series

    frame = frame.drop(columns=["Index"])
    return restore_index(frame, orig_index)


def contig_prime_modulo(data, n, per_day_normalize=False, verbose=False):
    frame, orig_index = _reset_with_work_index(data)
    primes = build_minimal_primes(n)
    if verbose and primes:
        print(f"utilizing {len(primes)} primes: {primes}")
    feature_map = _compute_contiguous_phase_features(
        frame=frame,
        value_col="RV_d",  # kept for signature consistency; function uses SR internally now
        primes=primes,
        n=n,
        per_day_normalize=per_day_normalize,
        prefix="CP",
    )
    _assign_phase_features(frame, feature_map)
    frame = frame.drop(columns=["Index"])
    return restore_index(frame, orig_index)


def random_sets(data, n, seed=None, fill_method="bfill"):
    frame, orig_index = _reset_with_work_index(data)
    horizon = int(n) if n else 0
    rng = np.random.default_rng(seed)
    primes = build_minimal_primes(n)
    created_cols = []
    for prime in primes:
        block_len = max(1, len(frame) // prime)
        offset = int(rng.integers(0, block_len)) if block_len > 1 else 0
        if "SR" not in frame.columns:
            raise KeyError("Expected column 'SR' (squared return / variance contribution). Add it in calculate_*_realized_volatility.")
        rolling = frame["SR"].rolling(window=block_len, min_periods=block_len)
        block_metric = np.sqrt(rolling.sum())
        mask = (frame["Index"] >= offset + block_len - 1) & (((frame["Index"] - offset + 1) % block_len) == 0)
        block_series = block_metric[mask].dropna()
        if block_series.empty:
            continue
        agg_window = max(1, horizon // block_len) if block_len else 1
        aggregated = block_series.rolling(window=agg_window, min_periods=1).mean()
        col_name = f"RV_rand_{block_len}_a{offset}"
        created_cols.append(col_name)
        frame[col_name] = np.nan
        frame.loc[aggregated.index, col_name] = aggregated
        frame[col_name] = frame[col_name].ffill()
        if fill_method == "bfill":
            frame[col_name] = frame[col_name].bfill()
        elif fill_method == "zero":
            frame[col_name] = frame[col_name].fillna(0.0)
        else:
            frame[col_name] = frame[col_name].bfill()
    if created_cols:
        frame[created_cols] = frame[created_cols].replace([np.inf, -np.inf], np.nan)
    frame = frame.drop(columns=["Index"])
    return restore_index(frame, orig_index)


def contig_prime_modulo_with_jumps(data, n, alpha=0.999, per_day_normalize=False, verbose=False):
    frame, orig_index = _reset_with_work_index(data)

    # Use variance contribution if available; otherwise fall back to RV_d.
    rv_var = frame["SR"] if "SR" in frame.columns else frame["RV_d"]

    const = np.sqrt(2 / np.pi)
    abs_returns = np.sqrt(rv_var).shift(1).abs()
    frame["BV_d"] = (const * abs_returns * np.sqrt(rv_var)).fillna(0.0)

    abs_returns_power = abs_returns ** (4 / 3)
    tq_product = abs_returns_power * abs_returns_power.shift(1) * abs_returns_power.shift(2)
    u_43 = 2 ** (2 / 3) * (np.pi ** (1 / 3)) / (4 ** (2 / 3) * (np.pi - 2) ** (1 / 3))
    frame["TQ_d"] = u_43 * (tq_product ** (3 / 4))

    delta = 1 / 252
    bv_sq = np.maximum(frame["BV_d"] ** 2, 1e-12)
    tq_ratio = frame["TQ_d"] / bv_sq
    variance_term = pd.Series(np.maximum(1.0, tq_ratio), index=frame.index).fillna(1.0)
    denom = np.sqrt(delta * ((np.pi ** 2) / 4 + np.pi - 5) * variance_term)
    frame["z_stat"] = ((rv_var - frame["BV_d"]) / denom).replace([np.inf, -np.inf], np.nan)

    critical_value = stats.norm.ppf(alpha)
    frame["jump_day"] = frame["z_stat"] > critical_value

    frame["J_d"] = np.where(frame["jump_day"], rv_var - frame["BV_d"], 0.0)
    frame["C_d"] = np.where(frame["jump_day"], frame["BV_d"], rv_var)

    primes = build_minimal_primes(n)
    if verbose and primes:
        print(f"utilizing {len(primes)} primes: {primes}")

    c_features = _compute_contiguous_phase_features(
        frame=frame,
        value_col="C_d",
        primes=primes,
        n=n,
        per_day_normalize=per_day_normalize,
        prefix="CP_C",
    )
    j_features = _compute_contiguous_phase_features(
        frame=frame,
        value_col="J_d",
        primes=primes,
        n=n,
        per_day_normalize=per_day_normalize,
        prefix="CP_J",
    )
    _assign_phase_features(frame, c_features)
    _assign_phase_features(frame, j_features)

    frame = frame.drop(columns=["Index"])
    return restore_index(frame, orig_index)


def contiguous_random_sets(data, n, seed=None, fill_method="bfill"):
    frame, orig_index = _reset_with_work_index(data)
    rng = np.random.default_rng(seed)
    horizon = int(n) if n else 0
    primes = build_minimal_primes(n)
    created_cols = []
    for prime in primes:
        offset = int(rng.integers(0, prime)) if prime > 1 else 0
        if "SR" not in frame.columns:
            raise KeyError("Expected column 'SR' (squared return / variance contribution). Add it in calculate_*_realized_volatility.")
        rolling = frame["SR"].rolling(window=prime, min_periods=prime)
        block_metric = np.sqrt(rolling.sum())
        mask = (frame["Index"] >= offset + prime - 1) & (((frame["Index"] - offset + 1) % prime) == 0)
        block_series = block_metric[mask].dropna()
        if block_series.empty:
            continue
        agg_window = max(1, horizon // prime)
        aggregated = block_series.rolling(window=agg_window, min_periods=1).mean()
        col_name = f"RV_contig_rand_{prime}_a{offset}"
        created_cols.append(col_name)
        frame[col_name] = np.nan
        frame.loc[aggregated.index, col_name] = aggregated
        frame[col_name] = frame[col_name].ffill()
        if fill_method == "bfill":
            frame[col_name] = frame[col_name].bfill()
        elif fill_method == "zero":
            frame[col_name] = frame[col_name].fillna(0.0)
        else:
            frame[col_name] = frame[col_name].bfill()
    if created_cols:
        frame[created_cols] = frame[created_cols].replace([np.inf, -np.inf], np.nan)
    frame = frame.drop(columns=["Index"])
    return restore_index(frame, orig_index)
