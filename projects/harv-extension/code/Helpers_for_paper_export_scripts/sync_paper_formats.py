from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs"
PAPER = OUT / "paper.tex"
QUANTFIN = OUT / "paper_quantfin_review.tex"


def replace_between(text: str, start: str, end: str, replacement: str) -> str:
    i = text.index(start)
    j = text.index(end, i) + len(end)
    return text[:i] + replacement + text[j:]


def update_original_tables(text: str) -> str:
    if "booktabs" not in text.split("\\begin{document}", 1)[0]:
        text = text.replace(
            "\\usepackage{graphicx, hyperref, setspace, titlesec, fancyhdr, multicol, parskip, indentfirst, etoolbox, caption, cite, xcolor}",
            "\\usepackage{graphicx, hyperref, setspace, titlesec, fancyhdr, multicol, parskip, indentfirst, etoolbox, caption, cite, xcolor, booktabs}",
        )

    text = text.replace(
        "\\captionsetup{font=small,hypcap=false}",
        "\\captionsetup{font=small,labelfont=bf,labelsep=period,hypcap=false}",
    )

    table_1a = r"""{\renewcommand{\thetable}{1A}
\begin{center}
\captionof{table}{Overall forecasting performance at the 5-minute horizon.}
\label{tab:overall_intraday}
\small
\renewcommand{\arraystretch}{1.15}
\setlength{\tabcolsep}{6pt}
\resizebox{0.98\textwidth}{!}{\input{paper_assets/tables_journal/table_overall_intraday.tex}}

\vspace{0.1cm}
\footnotesize \parbox{0.92\textwidth}{\textit{Notes:} Lower values are better except for Directional Accuracy. Entries report the cross-sectional mean with standard errors across assets. Error columns are rescaled for readability. Each asset contributes roughly 36,747 aligned 5-minute evaluation observations. Bold marks the best value in each column.}
\end{center}}"""
    text = replace_between(
        text,
        "{\\renewcommand{\\thetable}{1A}",
        "\\end{center}}",
        table_1a,
    )

    table_2 = r"""{\renewcommand{\thetable}{2}
\begin{center}
\captionof{table}{SMAPE by realized-volatility regime.}
\label{tab:regimes_intraday}
\small
\renewcommand{\arraystretch}{1.15}
\setlength{\tabcolsep}{10pt}
\input{paper_assets/tables_journal/table_regimes_intraday.tex}

\vspace{0.1cm}
\footnotesize \parbox{0.84\textwidth}{\textit{Notes:} Lower values are better. Regimes are defined by within-asset terciles of realized volatility. Entries are cross-sectional mean SMAPE with standard errors across assets.}
\end{center}}"""
    text = replace_between(
        text,
        "{\\renewcommand{\\thetable}{2}",
        "\\end{center}}",
        table_2,
    )

    text = text.replace(
        "\\resizebox{0.98\\textwidth}{!}{\\input{paper_assets/tables/Section_5_table_3_robust_summary.tex}}",
        "\\resizebox{0.98\\textwidth}{!}{\\input{paper_assets/tables_journal/table_section5_summary.tex}}",
    )
    text = text.replace(
        "\\resizebox{0.98\\textwidth}{!}{\\input{paper_assets/tables/Section_5_table_4_driver_checks.tex}}",
        "\\resizebox{0.98\\textwidth}{!}{\\input{paper_assets/tables_journal/table_section5_drivers.tex}}",
    )
    text = text.replace(
        "rather than merely benefiting from arbitrary feature enrichment.\n\n{\\renewcommand{\\thetable}{4}",
        "rather than merely benefiting from arbitrary feature enrichment.\n\n\\clearpage\n\n{\\renewcommand{\\thetable}{4}",
    )
    text = text.replace(
        "feature enrichment.\n\n\\clearpage\n\n\\clearpage\n\n{\\renewcommand{\\thetable}{4}",
        "feature enrichment.\n\n\\clearpage\n\n{\\renewcommand{\\thetable}{4}",
    )

    table_5 = r"""{\renewcommand{\thetable}{5}
\begin{center}
\captionof{table}{Group-level performance and significance by asset class.}
\label{tab:groups}
\small
\renewcommand{\arraystretch}{1.15}
\setlength{\tabcolsep}{8pt}
\resizebox{0.88\textwidth}{!}{\input{paper_assets/tables_journal/table_asset_groups.tex}}

\vspace{0.1cm}
\footnotesize \parbox{0.92\textwidth}{\textit{Notes:} Positive $\Delta \mathrm{SMAPE}$ indicates lower SMAPE than HAR-RV. Asset Win Rate is the share of assets in each group with positive $\Delta \mathrm{SMAPE}$ relative to HAR-RV. Fisher combined $p$-values aggregate per-asset Diebold--Mariano tests based on absolute-error loss. Values below machine precision are reported as $< 10^{-300}$. Standard errors are unavailable for single-asset groups.}
\end{center}}"""
    text = text.replace(
        "\n\n{\\renewcommand{\\thetable}{5}\n\\begin{center}",
        "\n\n\\clearpage\n\n{\\renewcommand{\\thetable}{5}\n\\begin{center}",
    )
    text = text.replace(
        "\\clearpage\n\n\\clearpage\n\n{\\renewcommand{\\thetable}{5}",
        "\\clearpage\n\n{\\renewcommand{\\thetable}{5}",
    )
    old_table_5_start = "\\noindent\\begin{minipage}{\\textwidth}\n{\\renewcommand{\\thetable}{5}"
    if old_table_5_start in text:
        text = replace_between(
            text,
            old_table_5_start,
            "\\end{minipage}",
            table_5,
        )
        text = text.replace(
            "\n\n{\\renewcommand{\\thetable}{5}\n\\begin{center}",
            "\n\n\\clearpage\n\n{\\renewcommand{\\thetable}{5}\n\\begin{center}",
        )

    table_1b = r"""{\renewcommand{\thetable}{1B}
\begin{center}
\captionof{table}{Overall forecasting performance at the daily horizon.}
\label{tab:overall_daily}
\small
\renewcommand{\arraystretch}{1.15}
\setlength{\tabcolsep}{6pt}
\resizebox{0.98\textwidth}{!}{\input{paper_assets/tables_journal/table_overall_daily.tex}}

\vspace{0.1cm}
\footnotesize \parbox{0.92\textwidth}{\textit{Notes:} Lower values are better except for Directional Accuracy. Entries report the cross-sectional mean with standard errors across assets. Error columns are rescaled for readability. Each asset contributes roughly 396 aligned daily observations. Bold marks the best value in each column.}
\end{center}}"""
    text = replace_between(
        text,
        "{\\renewcommand{\\thetable}{1B}",
        "\\end{center}}",
        table_1b,
    )

    appendix_inputs = {
        "Appendix_Table_C1_feature_count.tex": "table_appendix_c1_feature_count.tex",
        "Appendix_Table_C2_contiguous_windows.tex": "table_appendix_c2_contiguous.tex",
        "Appendix_Table_C3_prime_nonprime.tex": "table_appendix_c3_prime_nonprime.tex",
        "Appendix_Table_C4_random_controls.tex": "table_appendix_c4_random.tex",
        "Appendix_Table_C5_external.tex": "table_appendix_c5_external.tex",
    }
    for old, new in appendix_inputs.items():
        text = text.replace(
            f"\\input{{paper_assets/tables/{old}}}",
            f"\\input{{paper_assets/tables_journal/{new}}}",
        )
    text = text.replace(
        "Positive $\\Delta$SMAPE means that the candidate model has lower SMAPE than HAR-RV on the aligned evaluation timestamps.\n\n{\\renewcommand{\\thetable}{C1}",
        "Positive $\\Delta$SMAPE means that the candidate model has lower SMAPE than HAR-RV on the aligned evaluation timestamps.\n\n\\clearpage\n\n{\\renewcommand{\\thetable}{C1}",
    )
    text = text.replace(
        "timestamps.\n\n\\clearpage\n\n\\clearpage\n\n{\\renewcommand{\\thetable}{C1}",
        "timestamps.\n\n\\clearpage\n\n{\\renewcommand{\\thetable}{C1}",
    )
    text = text.replace(
        "keeping the exact CP construction as one compact implementation.\n\n{\\renewcommand{\\thetable}{C3}",
        "keeping the exact CP construction as one compact implementation.\n\n\\clearpage\n\n{\\renewcommand{\\thetable}{C3}",
    )
    text = text.replace(
        "implementation.\n\n\\clearpage\n\n\\clearpage\n\n{\\renewcommand{\\thetable}{C3}",
        "implementation.\n\n\\clearpage\n\n{\\renewcommand{\\thetable}{C3}",
    )
    text = remove_asset_group_leftover(text)
    return text


def remove_asset_group_leftover(text: str) -> str:
    marker = "\\vspace{0.25cm}\n\n\\begin{minipage}{0.88\\textwidth}\n\\centering\n\\textbf{Panel B. Contiguous Prime (CP)}"
    if marker not in text:
        return text
    start = text.index(marker)
    next_text = "\n\n\\noindent\nFigure~\\ref{fig:asset_groups}"
    end = text.index(next_text, start)
    return text[:start] + text[end + 2 :]


def extract_original_author_block(text: str) -> str:
    start = text.index("% Authors Block")
    end = text.index("\\singlespacing", start)
    return text[start:end].strip()


def build_quantfin_from_original(original_text: str) -> str:
    author_block = extract_original_author_block(original_text)
    body_start = original_text.index("\\begin{multicols}{2}")
    body_end = original_text.index("\\end{document}")
    body = original_text[body_start:body_end].strip()
    body = body.replace("\\begin{multicols}{2}\n\\setlength{\\columnsep}{0.5cm}\n\n", "")
    body = body.replace("\n\\end{multicols}\n\n\n\\section{Results and Analysis}", "\n\n\\section{Results and Analysis}")
    body = body.replace("paper_assets/figures/", "paper_assets/figures_journal/")
    body = body.replace(
        "while equal-window and prefix-window controls remain close to CP.\n\n{\\renewcommand{\\thetable}{C2}",
        "while equal-window and prefix-window controls remain close to CP.\n\n\\clearpage\n\n{\\renewcommand{\\thetable}{C2}",
    )

    preamble = r"""\documentclass[11pt,a4paper]{article}

\usepackage[margin=1in]{geometry}
\usepackage{setspace}
\usepackage{amsmath,amssymb,amsthm}
\usepackage{graphicx}
\usepackage{subcaption}
\usepackage{booktabs}
\usepackage{array}
\usepackage{caption}
\usepackage{multicol}
\usepackage{xcolor}
\definecolor{darkblue}{rgb}{0.0,0.0,0.55}
\usepackage[colorlinks=true,linkcolor=darkblue,citecolor=darkblue,urlcolor=darkblue]{hyperref}

\onehalfspacing
\setlength{\parindent}{0.5in}
\setlength{\parskip}{0pt}
\captionsetup[table]{font=small,labelfont=bf,labelsep=period,justification=centering,singlelinecheck=false,skip=6pt}
\captionsetup[figure]{font=small,labelfont=bf,labelsep=period,justification=justified,singlelinecheck=true,skip=6pt}
\newtheorem{theorem}{Theorem}

\title{\textbf{Using Prime Modulo Classes to Improve the HAR-RV Model}}
\author{}
\date{March 15, 2025}

\renewcommand{\thesection}{\Roman{section}.}
\renewcommand{\thesubsection}{\textit{\Alph{subsection}.}}
\renewcommand{\thesubsubsection}{\textit{\arabic{subsubsection}.}}

\begin{document}

\maketitle
\vspace{-0.4cm}
"""
    return preamble + "\n" + author_block + "\n\n" + body + "\n\\end{document}\n"


def main() -> None:
    original = PAPER.read_text(encoding="utf-8")
    updated_original = update_original_tables(original)
    PAPER.write_text(updated_original, encoding="utf-8", newline="\n")
    QUANTFIN.write_text(build_quantfin_from_original(updated_original), encoding="utf-8", newline="\n")


if __name__ == "__main__":
    main()
