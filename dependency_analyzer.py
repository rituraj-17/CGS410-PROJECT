# CGS410 Project: Dependency Tree Structure Analysis
# Pipeline for extracting syntactic metrics from SUD treebanks and LLM-generated text.

import argparse
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

# Suppress matplotlib config warnings in restricted environments
os.environ.setdefault("MPLCONFIGDIR", str((Path(".mplconfig")).resolve()))
os.environ.setdefault("MPLBACKEND", "Agg")
DEFAULT_STANZA_DIR = (Path(os.environ.get("STANZA_RESOURCES_DIR", ".stanza_resources"))).resolve()

import pandas as pd
import matplotlib.pyplot as plt
try:
    import seaborn as sns
    _HAVE_SEABORN = True
except ImportError:
    _HAVE_SEABORN = False

from conllu import parse_incr
import stanza


@dataclass(frozen=True)
class SentenceMetrics:
    tree_depth: float
    mean_node_arity: float
    dependency_distances: List[int]


def _safe_mean(values: Sequence[float]) -> float:
    return float(sum(values) / len(values)) if values else float("nan")

def _is_real_token(token: dict) -> bool:
    return isinstance(token.get("id"), int)

def _token_is_punct(token: dict) -> bool:
    return token.get("upos") == "PUNCT"


def compute_metrics_from_heads(
    heads_by_index: Dict[int, int],
    *,
    exclude_punct: bool = False,
    punct_by_index: Optional[Dict[int, bool]] = None,
) -> SentenceMetrics:
    
    if exclude_punct:
        if punct_by_index is None:
            raise ValueError("exclude_punct=True requires punct_by_index")
        kept = {i for i in heads_by_index.keys() if not punct_by_index.get(i, False)}
    else:
        kept = set(heads_by_index.keys())

    children: Dict[int, List[int]] = {i: [] for i in kept}
    roots: List[int] = []
    dependency_distances: List[int] = []

    for dep_i, head_i in heads_by_index.items():
        if dep_i not in kept:
            continue
        if exclude_punct and (head_i not in kept) and head_i != 0:
            continue
            
        if head_i == 0:
            roots.append(dep_i)
        elif head_i in kept:
            children[head_i].append(dep_i)
            dependency_distances.append(abs(head_i - dep_i))

    arities = [len(children[i]) for i in kept]
    mean_node_arity = _safe_mean([float(a) for a in arities])

    def depth_from(node: int) -> int:
        if not children[node]: return 1
        return 1 + max(depth_from(c) for c in children[node])

    tree_depth = float(max(depth_from(r) for r in roots)) if roots else float("nan")

    return SentenceMetrics(tree_depth, mean_node_arity, dependency_distances)


def compute_metrics_from_conllu_sentence(sentence_tokens: List[dict], *, exclude_punct: bool = False) -> SentenceMetrics:
    heads_by_index = {}
    punct_by_index = {}

    for tok in sentence_tokens:
        if not _is_real_token(tok): continue
        idx = int(tok["id"])
        head = tok.get("head")
        if head is None: continue
        
        heads_by_index[idx] = int(head)
        punct_by_index[idx] = _token_is_punct(tok)

    return compute_metrics_from_heads(heads_by_index, exclude_punct=exclude_punct, punct_by_index=punct_by_index)


def compute_metrics_from_stanza_sentence(stanza_sentence, *, exclude_punct: bool = False) -> SentenceMetrics:
    heads_by_index = {}
    punct_by_index = {}

    for w in stanza_sentence.words:
        idx = int(w.id)
        heads_by_index[idx] = int(w.head)
        punct_by_index[idx] = (w.upos == "PUNCT")

    return compute_metrics_from_heads(heads_by_index, exclude_punct=exclude_punct, punct_by_index=punct_by_index)


def iter_conllu_sentences(paths: Sequence[Path]) -> Iterable[List[dict]]:
    for p in paths:
        with p.open("r", encoding="utf-8") as f:
            for tokenlist in parse_incr(f):
                yield list(tokenlist)


def analyze_sud_treebank(*, conllu_paths: Sequence[Path], language: str, exclude_punct: bool) -> Tuple[pd.DataFrame, pd.DataFrame]:
    sent_rows, arc_rows = [], []
    
    for sent_idx, tokens in enumerate(iter_conllu_sentences(conllu_paths), start=1):
        sent_id = f"{language}-human-{sent_idx:07d}"
        m = compute_metrics_from_conllu_sentence(tokens, exclude_punct=exclude_punct)

        for d in m.dependency_distances:
            arc_rows.append({"dataset": "human", "language": language, "sent_id": sent_id, "dependency_distance": int(d)})

        real_tokens = [t for t in tokens if _is_real_token(t)]
        if exclude_punct:
            real_tokens = [t for t in real_tokens if not _token_is_punct(t)]

        sent_rows.append({
            "dataset": "human", "language": language, "sent_id": sent_id,
            "tree_depth": m.tree_depth, "mean_node_arity": m.mean_node_arity,
            "mean_dependency_distance": _safe_mean([float(x) for x in m.dependency_distances]),
            "n_tokens": len(real_tokens), "n_arcs": len(m.dependency_distances)
        })

    return pd.DataFrame(sent_rows), pd.DataFrame(arc_rows)


def load_llm_sentences(path: Path) -> List[str]:
    sentences = []
    if path.suffix.lower() == ".jsonl":
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip(): continue
                txt = str(json.loads(line.strip()).get("text", "")).strip()
                if txt: sentences.append(txt)
    else:
        with path.open("r", encoding="utf-8") as f:
            sentences = [line.strip() for line in f if line.strip()]
    return sentences


def build_stanza_pipeline(language: str, *, use_gpu: bool = False):
    DEFAULT_STANZA_DIR.mkdir(parents=True, exist_ok=True)
    try:
        stanza.download(language, processors="tokenize,pos,lemma,depparse", verbose=False, model_dir=str(DEFAULT_STANZA_DIR))
    except Exception:
        pass # Suppress verbose error, rely on pipeline init
        
    return stanza.Pipeline(lang=language, processors="tokenize,pos,lemma,depparse", tokenize_pretokenized=False, use_gpu=use_gpu, dir=str(DEFAULT_STANZA_DIR), verbose=False)


def analyze_llm_sentences(*, sentences: Sequence[str], language: str, exclude_punct: bool, use_gpu: bool, batch_size: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if not sentences:
        return pd.DataFrame(), pd.DataFrame()

    nlp = build_stanza_pipeline(language, use_gpu=use_gpu)
    sent_rows, arc_rows = [], []

    for start in range(0, len(sentences), batch_size):
        doc = nlp("\n\n".join(sentences[start : start + batch_size]))
        
        for s in doc.sentences:
            sent_id = f"{language}-llm-{(len(sent_rows) + 1):07d}"
            m = compute_metrics_from_stanza_sentence(s, exclude_punct=exclude_punct)

            for d in m.dependency_distances:
                arc_rows.append({"dataset": "llm", "language": language, "sent_id": sent_id, "dependency_distance": int(d)})

            words = [w for w in s.words if w.upos != "PUNCT"] if exclude_punct else s.words
            
            sent_rows.append({
                "dataset": "llm", "language": language, "sent_id": sent_id,
                "tree_depth": m.tree_depth, "mean_node_arity": m.mean_node_arity,
                "mean_dependency_distance": _safe_mean([float(x) for x in m.dependency_distances]),
                "n_tokens": len(words), "n_arcs": len(m.dependency_distances)
            })

    return pd.DataFrame(sent_rows), pd.DataFrame(arc_rows)


def plot_dependency_distance_boxplot(df_long: pd.DataFrame, *, output_path: Path, title: str) -> None:
    if _HAVE_SEABORN:
        sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)
    fig, ax = plt.subplots(figsize=(8, 5), dpi=150)

    if _HAVE_SEABORN:
        sns.boxplot(data=df_long, x="dataset", hue="dataset", y="dependency_distance", ax=ax, width=0.6, showfliers=False, palette={"human": "#1f77b4", "llm": "#ff7f0e"}, legend=False)
    else:
        values = [df_long.loc[df_long["dataset"] == g, "dependency_distance"].dropna().values for g in ["human", "llm"]]
        ax.boxplot(values, labels=["Human", "LLM"], showfliers=False)

    ax.set_title(title)
    ax.set_ylabel("Dependency distance")
    ax.set_xticklabels(["Human", "LLM"])
    for spine in ("top", "right"): ax.spines[spine].set_visible(False)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def plot_tree_depth_by_language_boxplot(df_sent: pd.DataFrame, *, output_path: Path, title: str) -> None:
    if df_sent.empty: return
    dfp = df_sent.loc[:, ["dataset", "language", "tree_depth"]].dropna(subset=["tree_depth"])
    
    if _HAVE_SEABORN: sns.set_theme(style="whitegrid", context="paper", font_scale=1.1)
    fig, ax = plt.subplots(figsize=(10, 5), dpi=150)

    if _HAVE_SEABORN:
        sns.boxplot(data=dfp, x="language", y="tree_depth", hue="dataset", ax=ax, showfliers=False, palette={"human": "#1f77b4", "llm": "#ff7f0e"})
        ax.legend(title="Dataset", loc="upper left")
    else:
        langs = sorted(dfp["language"].unique().tolist())
        positions, values, xticks, x = [], [], [], 1
        for lang in langs:
            for ds in ["human", "llm"]:
                positions.append(x)
                values.append(dfp.loc[(dfp["language"] == lang) & (dfp["dataset"] == ds), "tree_depth"].values)
                x += 1
            xticks.append(lang)
            x += 1
        ax.boxplot(values, positions=positions, showfliers=False)
        ax.set_xticks([p + 0.5 for p in range(1, len(langs) * 3, 3)], xticks)

    ax.set_title(title)
    ax.set_ylabel("Tree depth (max root-to-leaf)")
    for spine in ("top", "right"): ax.spines[spine].set_visible(False)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def plot_length_vs_distance_scatter(df_sent: pd.DataFrame, *, output_path: Path, title: str) -> None:
    dfp = df_sent.loc[:, ["dataset", "n_tokens", "mean_dependency_distance"]].replace([math.inf, -math.inf], math.nan).dropna()
    if dfp.empty: return

    if not _HAVE_SEABORN: return # Skip if seaborn unavailable

    sns.set_theme(style="whitegrid", context="paper", font_scale=1.1)
    g = sns.lmplot(data=dfp, x="n_tokens", y="mean_dependency_distance", hue="dataset", height=5, aspect=1.4, scatter_kws={"s": 10, "alpha": 0.35}, line_kws={"lw": 2}, palette={"human": "#1f77b4", "llm": "#ff7f0e"})
    g.set_axis_labels("Number of tokens", "Mean dependency distance")
    g.fig.suptitle(title)
    g.fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    g.fig.savefig(output_path, bbox_inches="tight")
    plt.close(g.fig)


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty: return df
    def _median(s): return float(s.median()) if len(s) else float("nan")

    return df.groupby(["dataset", "language"], dropna=False).agg(
        n_sentences=("sent_id", "count"),
        tree_depth_mean=("tree_depth", "mean"),
        tree_depth_median=("tree_depth", _median),
        node_arity_mean=("mean_node_arity", "mean"),
        node_arity_median=("mean_node_arity", _median),
        dep_dist_mean=("mean_dependency_distance", "mean"),
        dep_dist_median=("mean_dependency_distance", _median),
        n_tokens_mean=("n_tokens", "mean"),
    ).reset_index()


def length_matched_summary(df_sent: pd.DataFrame, *, min_tokens: int = 10, max_tokens: int = 20) -> pd.DataFrame:
    if df_sent.empty: return pd.DataFrame()
    return summarize(df_sent.loc[(df_sent["n_tokens"] >= min_tokens) & (df_sent["n_tokens"] <= max_tokens)].copy())


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Dependency tree metrics analysis.")
    parser.add_argument("--sud", nargs=2, action="append", metavar=("LANG", "GLOB"))
    parser.add_argument("--llm", nargs=2, action="append", metavar=("LANG", "FILE"))
    parser.add_argument("--exclude-punct", action="store_true")
    parser.add_argument("--use-gpu", action="store_true")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--out-csv", type=str, default="outputs/metrics_sentences.csv")
    parser.add_argument("--out-summary-csv", type=str, default="outputs/metrics_summary.csv")
    parser.add_argument("--out-plot", type=str, default="outputs/dep_distance_boxplot.png")
    parser.add_argument("--out-arcs-csv", type=str, default="outputs/dep_distance_arcs.csv")
    parser.add_argument("--plot-title", type=str, default="Dependency distance: Humans vs LLMs")
    parser.add_argument("--out-tree-depth-plot", type=str, default="outputs/tree_depth_boxplot.png")
    parser.add_argument("--out-length-vs-distance-plot", type=str, default="outputs/length_vs_distance_scatter.png")
    parser.add_argument("--out-length-matched-csv", type=str, default="outputs/metrics_length_matched.csv")
    args = parser.parse_args(argv)
    if not args.sud and not args.llm: parser.error("Must provide --sud or --llm")
    return args

def expand_glob(pattern: str) -> List[Path]:
    matches = sorted(Path().glob(pattern))
    if not matches: raise FileNotFoundError(f"No match for: {pattern}")
    return matches

def main(argv=None):
    args = parse_args(argv)
    all_sentence_dfs, all_arc_dfs = [], []

    if args.sud:
        for lang, glob_pat in args.sud:
            df_sent, df_arcs = analyze_sud_treebank(conllu_paths=expand_glob(glob_pat), language=lang, exclude_punct=args.exclude_punct)
            all_sentence_dfs.append(df_sent)
            all_arc_dfs.append(df_arcs)

    if args.llm:
        for lang, file_path in args.llm:
            df_sent, df_arcs = analyze_llm_sentences(sentences=load_llm_sentences(Path(file_path)), language=lang, exclude_punct=args.exclude_punct, use_gpu=args.use_gpu, batch_size=args.batch_size)
            all_sentence_dfs.append(df_sent)
            all_arc_dfs.append(df_arcs)

    df = pd.concat(all_sentence_dfs, ignore_index=True) if all_sentence_dfs else pd.DataFrame()
    arcs_df = pd.concat(all_arc_dfs, ignore_index=True) if all_arc_dfs else pd.DataFrame()

    for out_path, data in [(args.out_csv, df), (args.out_arcs_csv, arcs_df)]:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        data.to_csv(out_path, index=False)

    df_summary = summarize(df)
    Path(args.out_summary_csv).parent.mkdir(parents=True, exist_ok=True)
    df_summary.to_csv(args.out_summary_csv, index=False)

    plot_dependency_distance_boxplot(arcs_df.loc[:, ["dataset", "language", "dependency_distance"]], output_path=Path(args.out_plot), title=args.plot_title)
    plot_tree_depth_by_language_boxplot(df, output_path=Path(args.out_tree_depth_plot), title="Tree depth by language: Humans vs LLMs")
    plot_length_vs_distance_scatter(df, output_path=Path(args.out_length_vs_distance_plot), title="Sentence length vs dependency distance")

    df_length_matched = length_matched_summary(df)
    Path(args.out_length_matched_csv).parent.mkdir(parents=True, exist_ok=True)
    df_length_matched.to_csv(args.out_length_matched_csv, index=False)

    with pd.option_context("display.max_rows", 200, "display.max_columns", 200):
        print("Summary:\n", df_summary)
        print("\nLength-matched (10-20 tokens):\n", df_length_matched)

    return 0

if __name__ == "__main__":
    raise SystemExit(main())



#python dependency_analyzer.py \
 # --sud en "_smoke/sud/en/*.conllu" \
  #--sud hi "_smoke/sud/hi/*.conllu" \
  #--sud de "_smoke/sud/de/*.conllu" \
  #--sud es "_smoke/sud/es/*.conllu" \
  # --llm en "_smoke/llm/en.txt" \
  #--llm hi "_smoke/llm/hi.txt" \
  #--llm de "_smoke/llm/de.txt" \
  #--llm es "_smoke/llm/es.txt"
