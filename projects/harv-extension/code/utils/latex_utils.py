import numpy as np
import pandas as pd
import re

LATEX_SPECIALS = {
    '&': r'\&', '%': r'\%', '$': r'\$', '#': r'\#',
    '_': r'\_', '{': r'\{', '}': r'\}', '~': r'\textasciitilde{}',
    '^': r'\textasciicircum{}', '\\': r'\textbackslash{}'
}

MARK_MAP = {'†': r'\dagger', '‡': r'\ddagger', '★': r'\star'}

def latex_escape(s: str) -> str:
    if s is None:
        return ''
    s = str(s)
    return ''.join(LATEX_SPECIALS.get(ch, ch) for ch in s)

def fmt_mean_se(mean, se, decimals=1, sci=False, unit=None, mark=''):
    if mean is None or (isinstance(mean, float) and not np.isfinite(mean)):
        return r'\textit{NA}'
    if se is None or (isinstance(se, float) and not np.isfinite(se)):
        se = 0.0
    if sci:
        m = f"{mean:.2e}"
        s = f"{se:.2e}"
    else:
        m = f"{mean:.{decimals}f}"
        s = f"{se:.{decimals}f}"
    unit_str = f"\\,{unit}" if unit else ""
    mark_str = f"$^{{{MARK_MAP.get(mark, '')}}}$" if mark in MARK_MAP else ""
    return f"{m}{unit_str} $\\pm$ {s}{unit_str}{mark_str}"

def fmt_number(x, decimals=3, sci=False):
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return r'\textit{NA}'
    if sci:
        return f"{x:.2e}"
    return f"{x:.{decimals}f}"

def pretty_model(code: str) -> str:
    # You can extend this mapping as needed
    m = {
        'HAR': 'HAR-RV',
        'HAR_J': 'HAR-RV-J',
        'HAR_CJ': 'HAR-RV-CJ',
        'HAR_TCJ': 'HAR-RV-TCJ',
        'PM': 'Prime Modulo',
        'PM_VOL': 'Prime Modulo (Vol Wtd)',
        'PM_ADAPT': 'Prime Modulo (Adaptive)',
        'CP': 'Contiguous Prime',
        'CP_CJ': 'Contiguous Prime (CJ-aware)',
        'EXH': 'Exhaustive Prefixes',
        'HAM': 'Hamming Codes',
        'RAND': 'Randomized Sets',
        'CRS': 'Contiguous Random Sets',
    }
    return m.get(str(code), str(code))
