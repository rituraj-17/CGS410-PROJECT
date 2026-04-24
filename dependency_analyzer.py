#!/usr/bin/env python3
"""
dependency_analyzer.py

Linguistics dependency-tree analytics pipeline:

- Load SUD treebank files in CoNLL-U format using `conllu`
- Parse LLM-generated sentences using `stanza` dependency parser
- Compute per-sentence metrics:
    1) Tree Depth
    2) Node Arity (average dependents per head)
    3) Dependency Distance (|head_index - dep_index|)
- Produce:
    - A Pandas DataFrame with per-sentence metrics + metadata
    - Summary stats grouped by dataset (Human vs LLM) and language
    - A clean box-plot comparing Dependency Distance distributions

This script is designed to be readable and appendix-friendly: it is heavily
commented and avoids hidden “magic” behavior.

Notes / assumptions:
- SUD data is provided as one or more `.conllu` files per language.
- LLM data is provided as a JSONL file with one sentence per line, or a plain
  text file with one sentence per line. You also specify the language code.
- Tree depth is computed on the directed dependency tree rooted at the token
  whose HEAD=0. If multiple roots exist (rare), we take the maximum depth among
  roots. If no explicit root exists, depth is set to NaN.
- Node arity is the mean number of dependents per token (including root token).
  Tokens that are punctuation can be optionally excluded.
- Dependency distance is computed for each dependency arc as the absolute
  difference between head and dependent token positions (1-indexed positions).
"""

from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

# -----------------------------
# Runtime environment defaults
# -----------------------------

# Matplotlib caches fonts and config under a user directory by default. In
# restricted environments that directory may not be writable, which can crash
# plotting (or at least spam warnings). Set this BEFORE importing matplotlib.
os.environ.setdefault("MPLCONFIGDIR", str((Path(".mplconfig")).resolve()))
# Force a headless backend so the script runs on servers/CI without a display.
os.environ.setdefault("MPLBACKEND", "Agg")

# Stanza downloads models to a user directory by default. For reproducibility
# (and to avoid permission problems), default to a project-local directory.
DEFAULT_STANZA_DIR = (Path(os.environ.get("STANZA_RESOURCES_DIR", ".stanza_resources"))).resolve()

import pandas as pd

# We support either seaborn or matplotlib styling; seaborn makes nicer defaults.
import matplotlib.pyplot as plt

try:
    import seaborn as sns  # type: ignore

    _HAVE_SEABORN = True
except Exception:
    sns = None  # type: ignore
    _HAVE_SEABORN = False

from conllu import parse_incr  # type: ignore

import stanza  # type: ignore


# -----------------------------
# Data structures and helpers
# -----------------------------


@dataclass(frozen=True)
class SentenceMetrics:
    """
    Container for metrics computed for a single sentence.

    We store:
    - `tree_depth`: maximum root-to-leaf depth (integer, or NaN if undefined)
    - `mean_node_arity`: average number of dependents per token (float)
    - `dependency_distances`: list of arc distances for that sentence (ints)
    """

    tree_depth: float
    mean_node_arity: float
    dependency_distances: List[int]


def _safe_mean(values: Sequence[float]) -> float:
    """Mean that returns NaN on empty input (instead of ZeroDivisionError)."""
    if not values:
        return float("nan")
    return float(sum(values) / len(values))


def _is_real_token(token: dict) -> bool:
    """
    Determine whether a CoNLL-U token is a “real” syntactic token.

    In CoNLL-U, some lines represent:
    - multiword tokens (ID like "1-2") -> should be excluded from dependency arcs
    - empty nodes (ID like "3.1") -> typically excluded for standard UD metrics

    The `conllu` library parses ID as int for regular tokens, or str/tuple for
    special cases. We keep only those whose ID is an int.
    """

    return isinstance(token.get("id"), int)


def _token_is_punct(token: dict) -> bool:
    """
    Identify punctuation tokens.

    UD conventions:
    - UPOS == "PUNCT" is the most reliable signal.
    - Some treebanks may have punctuation in XPOS; we prioritize UPOS.
    """

    return token.get("upos") == "PUNCT"


# -----------------------------
# Core metric computations
# -----------------------------


def compute_metrics_from_heads(
    heads_by_index: Dict[int, int],
    *,
    exclude_punct: bool = False,
    punct_by_index: Optional[Dict[int, bool]] = None,
) -> SentenceMetrics:
    """
    Compute tree depth, node arity, and dependency distance given HEAD pointers.

    Parameters:
    - heads_by_index: mapping from token index i (1..N) to head index h (0..N)
      where 0 denotes ROOT.
    - exclude_punct: if True, tokens marked as punctuation are excluded from:
        - the arity denominator and counts,
        - the dependency distance arcs,
        - the depth computation (effectively pruning punctuation from tree).
      This can be useful if you want metrics to reflect syntactic structure
      rather than orthography.
    - punct_by_index: optional mapping i -> True/False indicating punctuation.

    Returns:
    - SentenceMetrics with:
        - tree_depth (float; integer-like, or NaN if no root)
        - mean_node_arity (float)
        - dependency_distances (list of ints)
    """

    # Decide which indices to keep based on punctuation.
    if exclude_punct:
        if punct_by_index is None:
            raise ValueError("exclude_punct=True requires punct_by_index")
        kept = {i for i in heads_by_index.keys() if not punct_by_index.get(i, False)}
    else:
        kept = set(heads_by_index.keys())

    # Build children lists for the kept nodes only.
    # We treat head=0 as ROOT; ROOT itself is not a token index.
    children: Dict[int, List[int]] = {i: [] for i in kept}
    roots: List[int] = []
    dependency_distances: List[int] = []

    for dep_i, head_i in heads_by_index.items():
        if dep_i not in kept:
            continue

        # If head is punctuation and we're excluding punct, treat as “no head”
        # (this is conservative; alternatives exist, but for reporting this is ok).
        if exclude_punct and (head_i not in kept) and head_i != 0:
            continue

        if head_i == 0:
            roots.append(dep_i)
        else:
            # Only link if both nodes are kept.
            if head_i in kept:
                children[head_i].append(dep_i)
                dependency_distances.append(abs(head_i - dep_i))

    # Node arity: number of dependents per (kept) token, averaged over tokens.
    arities = [len(children[i]) for i in kept]
    mean_node_arity = _safe_mean([float(a) for a in arities])

    # Tree depth: maximum root-to-leaf depth.
    # Depth convention: root token depth=1, its child depth=2, ...
    def depth_from(node: int) -> int:
        if not children[node]:
            return 1
        return 1 + max(depth_from(c) for c in children[node])

    if not roots:
        tree_depth = float("nan")
    else:
        tree_depth = float(max(depth_from(r) for r in roots))

    return SentenceMetrics(
        tree_depth=tree_depth,
        mean_node_arity=mean_node_arity,
        dependency_distances=dependency_distances,
    )


def compute_metrics_from_conllu_sentence(
    sentence_tokens: List[dict],
    *,
    exclude_punct: bool = False,
) -> SentenceMetrics:
    """
    Compute metrics for one sentence represented as a list of CoNLL-U token dicts
    (as returned by `conllu.parse_incr`).
    """

    # Extract HEAD pointers for real tokens only.
    heads_by_index: Dict[int, int] = {}
    punct_by_index: Dict[int, bool] = {}

    for tok in sentence_tokens:
        if not _is_real_token(tok):
            continue
        idx = int(tok["id"])
        head = tok.get("head")
        # HEAD can be None in malformed inputs; skip those tokens.
        if head is None:
            continue
        heads_by_index[idx] = int(head)
        punct_by_index[idx] = _token_is_punct(tok)

    return compute_metrics_from_heads(
        heads_by_index,
        exclude_punct=exclude_punct,
        punct_by_index=punct_by_index,
    )


def compute_metrics_from_stanza_sentence(
    stanza_sentence: "stanza.models.common.doc.Sentence",
    *,
    exclude_punct: bool = False,
) -> SentenceMetrics:
    """
    Compute metrics for a single sentence produced by stanza.

    Stanza tokens are 1-indexed in their `id` field (for words).
    Each word has a `.head` attribute where 0 denotes root.
    """

    heads_by_index: Dict[int, int] = {}
    punct_by_index: Dict[int, bool] = {}

    # IMPORTANT: In stanza, `sentence.words` corresponds to syntactic words;
    # `sentence.tokens` corresponds to orthographic tokens (and can group words).
    # For dependency parsing metrics, we use `words`.
    for w in stanza_sentence.words:
        idx = int(w.id)
        heads_by_index[idx] = int(w.head)
        punct_by_index[idx] = (w.upos == "PUNCT")

    return compute_metrics_from_heads(
        heads_by_index,
        exclude_punct=exclude_punct,
        punct_by_index=punct_by_index,
    )


# -----------------------------
# Loading human (SUD) data
# -----------------------------


def iter_conllu_sentences(paths: Sequence[Path]) -> Iterable[List[dict]]:
    """
    Stream sentences from one or more `.conllu` files.

    We use incremental parsing (`parse_incr`) to avoid loading entire treebanks
    into memory, which can be important for large corpora.
    """

    for p in paths:
        with p.open("r", encoding="utf-8") as f:
            for tokenlist in parse_incr(f):
                # `tokenlist` is a conllu.TokenList (list-like of dict tokens).
                yield list(tokenlist)


def analyze_sud_treebank(
    *,
    conllu_paths: Sequence[Path],
    language: str,
    exclude_punct: bool,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Analyze a set of SUD `.conllu` files for one language.

    Returns two DataFrames:

    1) Sentence-level DataFrame with one row per sentence and the following columns:
    - dataset: "human"
    - language
    - sent_id: sentence identifier if present (else None)
    - tree_depth
    - mean_node_arity
    - mean_dependency_distance (mean over arcs within sentence)
    - n_tokens (count of real tokens, optionally excluding punct)
    - n_arcs (number of dependency arcs used in distance computation)

    2) Arc-level DataFrame with one row per dependency arc and the following columns:
    - dataset: "human"
    - language
    - sent_id
    - dependency_distance
    """

    sent_rows = []
    arc_rows = []
    for sent_idx, tokens in enumerate(iter_conllu_sentences(conllu_paths), start=1):
        # Sentence metadata: conllu stores it in a TokenList object; since we
        # converted to list(dict), metadata is not directly carried. We can
        # recover it by re-parsing as TokenList if needed, but that would be
        # slower. Instead, we extract common IDs from comment lines ourselves:
        # in this script we keep a simple generated sent_id.
        #
        # If you need true treebank sent_id, a robust extension is:
        #   - keep TokenList rather than list(tokenlist)
        #   - use tokenlist.metadata.get("sent_id")
        sent_id = f"{language}-human-{sent_idx:07d}"

        # Compute metrics.
        m = compute_metrics_from_conllu_sentence(tokens, exclude_punct=exclude_punct)

        # Arc-level distances (one row per arc). This is what we use for the
        # Humans vs LLMs dependency-distance boxplot.
        for d in m.dependency_distances:
            arc_rows.append(
                {
                    "dataset": "human",
                    "language": language,
                    "sent_id": sent_id,
                    "dependency_distance": int(d),
                }
            )

        # Token counts (real tokens only).
        real_tokens = [t for t in tokens if _is_real_token(t)]
        if exclude_punct:
            real_tokens = [t for t in real_tokens if not _token_is_punct(t)]

        sent_rows.append(
            {
                "dataset": "human",
                "language": language,
                "sent_id": sent_id,
                "tree_depth": m.tree_depth,
                "mean_node_arity": m.mean_node_arity,
                "mean_dependency_distance": _safe_mean([float(x) for x in m.dependency_distances]),
                "n_tokens": len(real_tokens),
                "n_arcs": len(m.dependency_distances),
            }
        )

    return pd.DataFrame(sent_rows), pd.DataFrame(arc_rows)


# -----------------------------
# Parsing + analyzing LLM data
# -----------------------------


def load_llm_sentences(path: Path) -> List[str]:
    """
    Load LLM-generated sentences from either:
    - `.jsonl` where each line is a JSON object containing a "text" field
      (common pattern in generation pipelines), OR
    - any other text file where each non-empty line is treated as a sentence.
    """

    sentences: List[str] = []
    suffix = path.suffix.lower()

    if suffix == ".jsonl":
        with path.open("r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                if "text" not in obj:
                    raise ValueError(f"JSONL line {line_no} missing required key 'text'")
                txt = str(obj["text"]).strip()
                if txt:
                    sentences.append(txt)
    else:
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                txt = line.strip()
                if txt:
                    sentences.append(txt)

    return sentences


def build_stanza_pipeline(language: str, *, use_gpu: bool = False) -> "stanza.Pipeline":
    """
    Create a stanza pipeline for dependency parsing.

    This function also ensures that the required stanza model is downloaded.
    """

    # For reproducibility and fewer surprises, we explicitly download.
    # If the model is already present, stanza will skip the download quickly.
    DEFAULT_STANZA_DIR.mkdir(parents=True, exist_ok=True)
    try:
        stanza.download(
            language,
            processors="tokenize,pos,lemma,depparse",
            verbose=False,
            model_dir=str(DEFAULT_STANZA_DIR),
        )
    except Exception as e:
        raise RuntimeError(
            "Failed to download stanza models. "
            "This usually means you are offline or behind a restricted proxy. "
            f"Try running once on an unrestricted network, or pre-download models into: {DEFAULT_STANZA_DIR} "
            "You can also set the environment variable STANZA_RESOURCES_DIR to choose a different writable location."
        ) from e

    return stanza.Pipeline(
        lang=language,
        processors="tokenize,pos,lemma,depparse",
        tokenize_pretokenized=False,
        use_gpu=use_gpu,
        dir=str(DEFAULT_STANZA_DIR),
        verbose=False,
    )


def analyze_llm_sentences(
    *,
    sentences: Sequence[str],
    language: str,
    exclude_punct: bool,
    use_gpu: bool,
    batch_size: int,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Parse and analyze LLM-generated sentences for a single language.

    We run stanza over the input sentences, then compute the same metrics as for
    human treebanks.

    Returns two DataFrames:

    1) Sentence-level DataFrame (one row per parsed sentence)
    2) Arc-level DataFrame (one row per dependency arc distance)
    """

    n_input = len(sentences)
    if n_input == 0:
        empty_sent = pd.DataFrame(
            columns=[
                "dataset",
                "language",
                "sent_id",
                "tree_depth",
                "mean_node_arity",
                "mean_dependency_distance",
                "n_tokens",
                "n_arcs",
            ]
        )
        empty_arcs = pd.DataFrame(columns=["dataset", "language", "sent_id", "dependency_distance"])
        return empty_sent, empty_arcs

    nlp = build_stanza_pipeline(language, use_gpu=use_gpu)

    sent_rows = []
    arc_rows = []
    # Batching: stanza can take a list of texts if you build the doc manually,
    # but the simplest stable approach is to parse in batches by concatenating
    # with blank lines. Stanza treats blank lines as sentence boundaries.
    #
    # We keep a mapping from global index to help generate IDs.
    for start in range(0, n_input, batch_size):
        batch = sentences[start : start + batch_size]
        doc_text = "\n\n".join(batch)
        doc = nlp(doc_text)

        # Stanza may split input lines into multiple sentences; that’s good and
        # linguistically realistic. We attribute each parsed sentence to the
        # batch. For traceability, we create sequential IDs.
        for s in doc.sentences:
            global_idx = len(sent_rows) + 1
            sent_id = f"{language}-llm-{global_idx:07d}"

            m = compute_metrics_from_stanza_sentence(s, exclude_punct=exclude_punct)

            for d in m.dependency_distances:
                arc_rows.append(
                    {
                        "dataset": "llm",
                        "language": language,
                        "sent_id": sent_id,
                        "dependency_distance": int(d),
                    }
                )

            # Token count: use stanza words (syntactic units).
            words = s.words
            if exclude_punct:
                words = [w for w in words if w.upos != "PUNCT"]

            sent_rows.append(
                {
                    "dataset": "llm",
                    "language": language,
                    "sent_id": sent_id,
                    "tree_depth": m.tree_depth,
                    "mean_node_arity": m.mean_node_arity,
                    "mean_dependency_distance": _safe_mean(
                        [float(x) for x in m.dependency_distances]
                    ),
                    "n_tokens": len(words),
                    "n_arcs": len(m.dependency_distances),
                }
            )

    return pd.DataFrame(sent_rows), pd.DataFrame(arc_rows)


# -----------------------------
# Plotting and reporting
# -----------------------------


def build_dependency_distance_long_df(arcs_df: pd.DataFrame) -> pd.DataFrame:
    """
    Build a “long” DataFrame suitable for boxplots of dependency distance.

    Input `arcs_df` should be arc-level, with one row per dependency arc:
    - dataset: "human" or "llm"
    - language
    - sent_id
    - dependency_distance
    """
    return arcs_df.loc[:, ["dataset", "language", "dependency_distance"]]


def plot_dependency_distance_boxplot(
    df_long: pd.DataFrame,
    *,
    output_path: Path,
    title: str,
) -> None:
    """
    Generate a clean box-plot comparing Humans vs LLMs dependency distance.

    The plot is saved to `output_path`.
    """

    # Style: prefer seaborn’s paper-like theme if available.
    if _HAVE_SEABORN:
        sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)
    else:
        plt.style.use("seaborn-v0_8-whitegrid")

    # We generate a grouped boxplot with dataset on x-axis.
    fig, ax = plt.subplots(figsize=(8, 5), dpi=150)

    if _HAVE_SEABORN:
        sns.boxplot(
            data=df_long,
            x="dataset",
            hue="dataset",
            y="dependency_distance",
            ax=ax,
            width=0.6,
            showfliers=False,
            palette={"human": "#1f77b4", "llm": "#ff7f0e"},
            legend=False,
        )
        sns.stripplot(
            data=df_long,
            x="dataset",
            y="dependency_distance",
            ax=ax,
            color="black",
            size=2,
            alpha=0.25,
            jitter=0.25,
        )
    else:
        # Matplotlib-only fallback: basic boxplot by dataset
        groups = ["human", "llm"]
        values = [
            df_long.loc[df_long["dataset"] == g, "dependency_distance"].dropna().values
            for g in groups
        ]
        ax.boxplot(values, labels=["Human", "LLM"], showfliers=False)

    ax.set_title(title)
    ax.set_xlabel("")
    ax.set_ylabel("Dependency distance (|head index − dependent index|)")
    ax.set_xticklabels(["Human", "LLM"])

    # Small aesthetic tweaks: remove top/right spines in seaborn style.
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def plot_tree_depth_by_language_boxplot(
    df_sent: pd.DataFrame,
    *,
    output_path: Path,
    title: str = "Tree depth by language (Humans vs LLMs)",
) -> None:
    """
    Grouped boxplot of per-sentence tree depth across languages.

    - x-axis: language
    - hue: dataset (human vs llm)
    - y-axis: tree_depth

    Note: Although the prompt asks for "tree_depth_mean", a boxplot is most
    informative when we show the *distribution* of sentence-level tree depths.
    The group means are still implicitly captured by the medians/boxes, and the
    numeric means remain available in `metrics_summary.csv`.
    """

    if df_sent.empty:
        return

    cols = {"dataset", "language", "tree_depth"}
    if not cols.issubset(set(df_sent.columns)):
        raise ValueError(f"df_sent missing required columns: {sorted(cols)}")

    dfp = df_sent.loc[:, ["dataset", "language", "tree_depth"]].dropna(subset=["tree_depth"])
    if dfp.empty:
        return

    if _HAVE_SEABORN:
        sns.set_theme(style="whitegrid", context="paper", font_scale=1.1)
    else:
        plt.style.use("seaborn-v0_8-whitegrid")

    fig, ax = plt.subplots(figsize=(10, 5), dpi=150)

    if _HAVE_SEABORN:
        sns.boxplot(
            data=dfp,
            x="language",
            y="tree_depth",
            hue="dataset",
            ax=ax,
            showfliers=False,
            palette={"human": "#1f77b4", "llm": "#ff7f0e"},
        )
        ax.legend(title="Dataset", loc="upper left", frameon=True)
    else:
        # Matplotlib-only fallback: draw separate boxplots per dataset-language.
        langs = sorted(dfp["language"].unique().tolist())
        datasets = ["human", "llm"]
        positions = []
        values = []
        xticks = []
        x = 1
        for lang in langs:
            for ds in datasets:
                positions.append(x)
                v = dfp.loc[(dfp["language"] == lang) & (dfp["dataset"] == ds), "tree_depth"].values
                values.append(v)
                x += 1
            xticks.append(lang)
            x += 1
        ax.boxplot(values, positions=positions, showfliers=False)
        ax.set_xticks([p + 0.5 for p in range(1, len(langs) * 3, 3)], xticks)

    ax.set_title(title)
    ax.set_xlabel("Language")
    ax.set_ylabel("Tree depth (max root-to-leaf depth)")
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def plot_length_vs_distance_scatter(
    df_sent: pd.DataFrame,
    *,
    output_path: Path,
    title: str = "Sentence length vs dependency distance",
) -> None:
    """
    Scatter plot with regression lines:

    - x-axis: n_tokens (sentence length)
    - y-axis: mean_dependency_distance (per-sentence mean arc distance)
    - hue: dataset (human vs llm)
    """

    if df_sent.empty:
        return

    cols = {"dataset", "n_tokens", "mean_dependency_distance"}
    if not cols.issubset(set(df_sent.columns)):
        raise ValueError(f"df_sent missing required columns: {sorted(cols)}")

    dfp = df_sent.loc[:, ["dataset", "n_tokens", "mean_dependency_distance"]].dropna()
    # Defensive: pathological rows (e.g., sentences with zero arcs) can yield
    # NaN/inf and destabilize the regression fit.
    dfp = dfp.replace([math.inf, -math.inf], math.nan).dropna()
    if dfp.empty:
        return

    if not _HAVE_SEABORN:
        # Minimal matplotlib-only fallback: scatter without regression.
        fig, ax = plt.subplots(figsize=(8, 5), dpi=150)
        for ds, c in (("human", "#1f77b4"), ("llm", "#ff7f0e")):
            sub = dfp.loc[dfp["dataset"] == ds]
            ax.scatter(sub["n_tokens"], sub["mean_dependency_distance"], s=10, alpha=0.35, label=ds, c=c)
        ax.legend(title="Dataset", frameon=True)
        ax.set_title(title)
        ax.set_xlabel("Number of tokens")
        ax.set_ylabel("Mean dependency distance (per sentence)")
        fig.tight_layout()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, bbox_inches="tight")
        plt.close(fig)
        return

    sns.set_theme(style="whitegrid", context="paper", font_scale=1.1)
    g = sns.lmplot(
        data=dfp,
        x="n_tokens",
        y="mean_dependency_distance",
        hue="dataset",
        height=5,
        aspect=1.4,
        scatter_kws={"s": 10, "alpha": 0.35},
        line_kws={"lw": 2},
        palette={"human": "#1f77b4", "llm": "#ff7f0e"},
    )
    g.set_axis_labels("Number of tokens", "Mean dependency distance (per sentence)")
    g.fig.suptitle(title)
    g.fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    g.fig.savefig(output_path, bbox_inches="tight")
    plt.close(g.fig)


def length_matched_summary(
    df_sent: pd.DataFrame,
    *,
    min_tokens: int = 10,
    max_tokens: int = 20,
) -> pd.DataFrame:
    """
    Filter to a length-matched subset and summarize the same core metrics.

    We keep only sentences where min_tokens <= n_tokens <= max_tokens, then run
    `summarize()` to compare Humans vs LLMs per language on that filtered data.
    """

    if df_sent.empty:
        return pd.DataFrame()

    required = {"n_tokens"}
    if not required.issubset(set(df_sent.columns)):
        raise ValueError(f"df_sent missing required columns: {sorted(required)}")

    df_filt = df_sent.loc[(df_sent["n_tokens"] >= min_tokens) & (df_sent["n_tokens"] <= max_tokens)].copy()
    return summarize(df_filt)


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    """
    Provide summary statistics by dataset and language.

    We report:
    - sentence count
    - mean and median for each metric
    """

    if df.empty:
        return df

    def _median(s: pd.Series) -> float:
        return float(s.median()) if len(s) else float("nan")

    grouped = df.groupby(["dataset", "language"], dropna=False)
    summary = grouped.agg(
        n_sentences=("sent_id", "count"),
        tree_depth_mean=("tree_depth", "mean"),
        tree_depth_median=("tree_depth", _median),
        node_arity_mean=("mean_node_arity", "mean"),
        node_arity_median=("mean_node_arity", _median),
        dep_dist_mean=("mean_dependency_distance", "mean"),
        dep_dist_median=("mean_dependency_distance", _median),
        n_tokens_mean=("n_tokens", "mean"),
    )
    return summary.reset_index()


# -----------------------------
# CLI wiring
# -----------------------------


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze SUD treebanks vs LLM-generated sentences using dependency metrics."
    )

    # Human SUD inputs: multiple (language, path_glob) pairs.
    # Example:
    #   --sud en "data/sud/en/*.conllu" --sud fr "data/sud/fr/*.conllu"
    parser.add_argument(
        "--sud",
        nargs=2,
        action="append",
        metavar=("LANG", "GLOB"),
        help="SUD inputs: language code and a glob for .conllu files. Can be repeated.",
        required=False,
    )

    # LLM input(s): multiple (language, file) pairs.
    # Example:
    #   --llm en llm_outputs/en.txt --llm fr llm_outputs/fr.jsonl
    parser.add_argument(
        "--llm",
        nargs=2,
        action="append",
        metavar=("LANG", "FILE"),
        help="LLM sentences: language code and file (.txt or .jsonl). Can be repeated.",
        required=False,
    )

    parser.add_argument(
        "--exclude-punct",
        action="store_true",
        help="Exclude punctuation tokens/arcs from all metrics.",
    )

    parser.add_argument(
        "--use-gpu",
        action="store_true",
        help="Use GPU for stanza if available (requires proper stanza/PyTorch setup).",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Batch size for stanza parsing (tradeoff between speed and memory).",
    )

    parser.add_argument(
        "--out-csv",
        type=str,
        default="outputs/metrics_sentences.csv",
        help="Where to save the per-sentence metrics CSV.",
    )
    parser.add_argument(
        "--out-summary-csv",
        type=str,
        default="outputs/metrics_summary.csv",
        help="Where to save the summary metrics CSV.",
    )
    parser.add_argument(
        "--out-plot",
        type=str,
        default="outputs/dep_distance_boxplot.png",
        help="Where to save the dependency distance boxplot image.",
    )
    parser.add_argument(
        "--out-arcs-csv",
        type=str,
        default="outputs/dep_distance_arcs.csv",
        help="Where to save the arc-level dependency distances CSV.",
    )
    parser.add_argument(
        "--plot-title",
        type=str,
        default="Dependency distance: Humans vs LLMs",
        help="Title for the dependency distance boxplot.",
    )
    parser.add_argument(
        "--out-tree-depth-plot",
        type=str,
        default="outputs/tree_depth_boxplot.png",
        help="Where to save the tree depth by language boxplot image.",
    )
    parser.add_argument(
        "--out-length-vs-distance-plot",
        type=str,
        default="outputs/length_vs_distance_scatter.png",
        help="Where to save the length-vs-distance scatter plot image.",
    )
    parser.add_argument(
        "--out-length-matched-csv",
        type=str,
        default="outputs/metrics_length_matched.csv",
        help="Where to save the length-matched (10-20 tokens) summary CSV.",
    )

    args = parser.parse_args(argv)
    if not args.sud and not args.llm:
        parser.error("You must provide at least one --sud or --llm input.")
    return args


def expand_glob(pattern: str) -> List[Path]:
    """Expand a glob pattern into a sorted list of Paths."""

    matches = sorted(Path().glob(pattern))
    if not matches:
        raise FileNotFoundError(f"No files matched glob: {pattern!r}")
    return matches


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)

    all_sentence_dfs: List[pd.DataFrame] = []
    all_arc_dfs: List[pd.DataFrame] = []

    # Human SUD analyses.
    if args.sud:
        for lang, glob_pat in args.sud:
            conllu_paths = expand_glob(glob_pat)
            df_human_sent, df_human_arcs = analyze_sud_treebank(
                conllu_paths=conllu_paths,
                language=lang,
                exclude_punct=args.exclude_punct,
            )
            all_sentence_dfs.append(df_human_sent)
            all_arc_dfs.append(df_human_arcs)

    # LLM analyses (dependency parsing via stanza).
    if args.llm:
        for lang, file_path in args.llm:
            llm_path = Path(file_path)
            sentences = load_llm_sentences(llm_path)
            df_llm_sent, df_llm_arcs = analyze_llm_sentences(
                sentences=sentences,
                language=lang,
                exclude_punct=args.exclude_punct,
                use_gpu=args.use_gpu,
                batch_size=args.batch_size,
            )
            all_sentence_dfs.append(df_llm_sent)
            all_arc_dfs.append(df_llm_arcs)

    # Combine both datasets into one sentence-level DataFrame.
    df = (
        pd.concat(all_sentence_dfs, ignore_index=True)
        if all_sentence_dfs
        else pd.DataFrame()
    )

    # Combine arc-level distances for boxplots.
    arcs_df = pd.concat(all_arc_dfs, ignore_index=True) if all_arc_dfs else pd.DataFrame()

    # Save per-sentence metrics.
    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)

    # Save arc-level distances (so you can re-plot / re-analyze without re-parsing).
    out_arcs = Path(args.out_arcs_csv)
    out_arcs.parent.mkdir(parents=True, exist_ok=True)
    arcs_df.to_csv(out_arcs, index=False)

    # Save summary metrics.
    df_summary = summarize(df)
    out_summary = Path(args.out_summary_csv)
    out_summary.parent.mkdir(parents=True, exist_ok=True)
    df_summary.to_csv(out_summary, index=False)

    # Plot dependency distance comparison (arc-level distances).
    df_long = build_dependency_distance_long_df(arcs_df)
    plot_dependency_distance_boxplot(
        df_long,
        output_path=Path(args.out_plot),
        title=args.plot_title,
    )

    # Additional analysis 1: Tree depth distribution by language (grouped by dataset).
    plot_tree_depth_by_language_boxplot(
        df,
        output_path=Path(args.out_tree_depth_plot),
        title="Tree depth by language: Humans vs LLMs",
    )

    # Additional analysis 2: Sentence length vs dependency distance with regression lines.
    plot_length_vs_distance_scatter(
        df,
        output_path=Path(args.out_length_vs_distance_plot),
        title="Sentence length vs dependency distance",
    )

    # Second analysis (length-matched): restrict to 10..20 tokens and re-summarize.
    df_length_matched_summary = length_matched_summary(df, min_tokens=10, max_tokens=20)
    out_len = Path(args.out_length_matched_csv)
    out_len.parent.mkdir(parents=True, exist_ok=True)
    df_length_matched_summary.to_csv(out_len, index=False)

    # Print a compact summary to stdout for quick inspection.
    with pd.option_context("display.max_rows", 200, "display.max_columns", 200):
        print(df_summary)
        print("\nLength-matched summary (10-20 tokens):")
        print(df_length_matched_summary)
    print(f"\nWrote per-sentence metrics to: {out_csv}")
    print(f"Wrote arc-level distances to:  {out_arcs}")
    print(f"Wrote summary metrics to:      {out_summary}")
    print(f"Wrote plot to:                {Path(args.out_plot)}")
    print(f"Wrote tree depth plot to:     {Path(args.out_tree_depth_plot)}")
    print(f"Wrote scatter plot to:        {Path(args.out_length_vs_distance_plot)}")
    print(f"Wrote length-matched CSV to:  {out_len}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
