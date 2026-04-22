from __future__ import annotations

import os
import re
import shutil
import tempfile
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
import yaml


ISSUE_COLUMNS = ["run_timestamp", "input_file", "severity", "issue_code", "count", "detail"]
URL_PATTERN = re.compile(r"(https?://\S+|www\.\S+)", flags=re.IGNORECASE)
SEGMENT_TOKEN_PATTERN = re.compile(r"[0-9A-Za-zÀ-ỹĐđ_]+")
TOKEN_PATTERN_TEXT_ONLY = re.compile(r"[a-zà-ỹđ_]+", flags=re.IGNORECASE)
TOKEN_PATTERN_WITH_NUMBER = re.compile(r"[0-9a-zà-ỹđ_]+", flags=re.IGNORECASE)
EMOJI_PATTERN = re.compile(
    "["
    "\U0001F300-\U0001F5FF"
    "\U0001F600-\U0001F64F"
    "\U0001F680-\U0001F6FF"
    "\U0001F700-\U0001F77F"
    "\U0001F780-\U0001F7FF"
    "\U0001F800-\U0001F8FF"
    "\U0001F900-\U0001F9FF"
    "\U0001FA00-\U0001FAFF"
    "\U00002700-\U000027BF"
    "]",
    flags=re.UNICODE,
)
SOCIAL_EXCLAMATION_PATTERN = re.compile(r"!{2,}")
SOCIAL_QUESTION_PATTERN = re.compile(r"\?{2,}")
SENTENCE_PUNCT_PATTERN = re.compile(r"[.!?;:]+")
HASHTAG_PATTERN = re.compile(r"#([^\s#]+)", flags=re.UNICODE)
HASHTAG_SANITIZE_PATTERN = re.compile(r"[^0-9A-Za-zÀ-ỹĐđ]+", flags=re.UNICODE)
HASHTAG_MULTI_UNDERSCORE_PATTERN = re.compile(r"_+")
SOCIAL_EMOJI_TOKEN = "EMOJI"
SOCIAL_HASHTAG_PREFIX = "HASHTAG_"
LEGACY_CONTROL_TOKENS = {
    "__eos__",
    "__multi_excl__",
    "__multi_q__",
    "eos",
    "multi_excl",
    "multi_q",
}
EMPTY_RATIO_WARNING_THRESHOLD = 0.20


class PrepareMlTextError(Exception):
    pass


# Load one YAML file and return a dictionary object.
def load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise PrepareMlTextError(f"Missing config file: {path}")
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise PrepareMlTextError(f"Config file must contain a mapping object: {path}")
    return data


# Append one standardized issue row to the issue list.
def add_issue(
    issues: list[dict[str, Any]],
    run_timestamp: str,
    input_file: str,
    severity: str,
    issue_code: str,
    count: int,
    detail: str,
) -> None:
    issues.append(
        {
            "run_timestamp": run_timestamp,
            "input_file": input_file,
            "severity": severity,
            "issue_code": issue_code,
            "count": int(count),
            "detail": detail,
        }
    )


# Determine per-file status based on issue severities.
def determine_status(source_issues: list[dict[str, Any]]) -> str:
    severities = {item["severity"] for item in source_issues}
    if "ERROR" in severities:
        return "FAIL"
    if "WARNING" in severities:
        return "WARNING"
    return "PASS"


# Ensure JAVA_HOME is available for py_vncorenlp/jnius bootstrap.
def ensure_java_home(warnings_out: list[str]) -> None:
    java_home = os.environ.get("JAVA_HOME", "").strip()
    if java_home and Path(java_home).exists():
        return

    java_root = Path("C:/Program Files/Java")
    if java_root.exists():
        candidates = sorted([p for p in java_root.iterdir() if p.is_dir() and p.name.lower().startswith("jdk")])
        if candidates:
            picked = candidates[-1]
            os.environ["JAVA_HOME"] = str(picked)
            warnings_out.append(f"auto_set_java_home:{picked.as_posix()}")
            return

    warnings_out.append("java_home_not_found")


# Copy VnCoreNLP model into a no-space runtime directory when needed.
def prepare_vncorenlp_runtime_dir(model_dir: Path | None, warnings_out: list[str]) -> Path | None:
    if model_dir is None:
        return None
    source_dir = model_dir.resolve()
    if not source_dir.exists():
        raise PrepareMlTextError(f"Configured segment_model_dir does not exist: {source_dir}")
    if " " not in str(source_dir):
        return source_dir

    runtime_dir = Path(tempfile.gettempdir()) / "vncorenlp_runtime_model"
    runtime_models = runtime_dir / "models"
    runtime_models.mkdir(parents=True, exist_ok=True)

    # Sync only required files for wseg annotator.
    required_files = [
        ("VnCoreNLP-1.2.jar", "VnCoreNLP-1.2.jar"),
        ("models/wordsegmenter/vi-vocab", "models/wordsegmenter/vi-vocab"),
        ("models/wordsegmenter/wordsegmenter.rdr", "models/wordsegmenter/wordsegmenter.rdr"),
    ]
    for rel_src, rel_dst in required_files:
        src = source_dir / rel_src
        dst = runtime_dir / rel_dst
        if not src.exists():
            raise PrepareMlTextError(f"Missing VnCoreNLP model file: {src}")
        dst.parent.mkdir(parents=True, exist_ok=True)
        if not dst.exists() or src.stat().st_mtime > dst.stat().st_mtime:
            shutil.copy2(src, dst)

    warnings_out.append(
        f"segment_model_dir_runtime_copy:{source_dir.as_posix()}->{runtime_dir.as_posix()}"
    )
    return runtime_dir


# Load Vietnamese stopwords from one token per line.
def load_stopwords(path: Path) -> set[str]:
    if not path.exists():
        raise PrepareMlTextError(f"Missing stopwords file: {path}")
    stopwords: set[str] = set()
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            token = line.strip().lower()
            if token and not token.startswith("#"):
                stopwords.add(token)
    return stopwords


# Build one segmentation callable based on configured backend.
def build_segmenter(cfg: dict[str, Any], warnings_out: list[str]) -> tuple[str, Any]:
    requested_backend = str(cfg["segment_backend"]).strip().lower()
    model_dir = cfg.get("segment_model_dir")

    # Fallback segmentation keeps only text-like tokens and preserves underscores.
    def regex_segment(text: str) -> str:
        return " ".join(SEGMENT_TOKEN_PATTERN.findall(text))

    if requested_backend not in {"auto", "regex", "py_vncorenlp"}:
        raise PrepareMlTextError(
            f"Unsupported ml_text.segment_backend='{requested_backend}'. "
            "Use one of: auto, regex, py_vncorenlp."
        )

    if requested_backend == "regex":
        return "regex", regex_segment

    # Try py_vncorenlp when explicitly requested or when auto mode can use it.
    ensure_java_home(warnings_out)
    runtime_model_dir = prepare_vncorenlp_runtime_dir(model_dir, warnings_out)
    try:
        import py_vncorenlp  # type: ignore

        init_kwargs: dict[str, Any] = {"annotators": ["wseg"]}
        if runtime_model_dir is not None:
            init_kwargs["save_dir"] = str(runtime_model_dir)
        segmenter = py_vncorenlp.VnCoreNLP(**init_kwargs)

        # Normalize py_vncorenlp output into one whitespace-tokenized string.
        def vncorenlp_segment(text: str) -> str:
            segmented = segmenter.word_segment(text)
            if isinstance(segmented, list):
                return " ".join(str(x) for x in segmented)
            return str(segmented)

        return "py_vncorenlp", vncorenlp_segment
    except Exception as exc:
        if requested_backend == "py_vncorenlp":
            raise PrepareMlTextError(
                "Failed to initialize py_vncorenlp backend. "
                f"Install/setup VnCoreNLP first. Details: {exc}"
            ) from exc
        warnings_out.append(f"segment_backend_fallback_regex:{exc}")
        return "regex", regex_segment


# Pre-clean raw text before segmentation.
def normalize_hashtag_token(match: re.Match[str]) -> str:
    tag_raw = match.group(1)
    tag_clean = HASHTAG_SANITIZE_PATTERN.sub("_", tag_raw).strip("_").lower()
    tag_clean = HASHTAG_MULTI_UNDERSCORE_PATTERN.sub("_", tag_clean)
    if not tag_clean:
        return " "
    return f" {SOCIAL_HASHTAG_PREFIX}{tag_clean} "


# Pre-clean raw text before segmentation.
def pre_clean_text(
    text: str,
    content_type: str,
    replace_underscore_with_space: bool,
    remove_url_tokens: bool,
    preserve_social_emoji: bool,
    remove_news_emoji: bool,
    preserve_social_punctuation_signals: bool,
    keep_sentence_boundary_token: bool,
) -> str:
    out = text
    if replace_underscore_with_space:
        out = out.replace("_", " ")
    if remove_url_tokens:
        out = URL_PATTERN.sub(" ", out)
    out = HASHTAG_PATTERN.sub(normalize_hashtag_token, out)

    # Keep emoji signal for social domain, remove icon/emoji noise for news domain.
    if content_type == "social":
        if preserve_social_emoji:
            out = EMOJI_PATTERN.sub(f" {SOCIAL_EMOJI_TOKEN} ", out)
    elif remove_news_emoji:
        out = EMOJI_PATTERN.sub(" ", out)

    # Keep punctuation markers out of output so only semantic tokens remain.
    out = SOCIAL_EXCLAMATION_PATTERN.sub(" ", out)
    out = SOCIAL_QUESTION_PATTERN.sub(" ", out)
    out = SENTENCE_PUNCT_PATTERN.sub(" ", out)

    out = re.sub(r"\s+", " ", out).strip()
    return out


# Convert one segmented text into filtered token sequence.
def tokenize_and_filter(
    text_segmented: str,
    lowercase: bool,
    keep_numbers: bool,
    min_token_length: int,
    remove_stopwords: bool,
    stopwords: set[str],
) -> tuple[str, int, int, list[str]]:
    token_pattern = TOKEN_PATTERN_WITH_NUMBER if keep_numbers else TOKEN_PATTERN_TEXT_ONLY
    text_variant = text_segmented.lower() if lowercase else text_segmented
    tokens = token_pattern.findall(text_variant)

    # Keep control tokens in canonical form across both text variants.
    if lowercase:
        normalized_tokens: list[str] = []
        hashtag_prefix_lower = SOCIAL_HASHTAG_PREFIX.lower()
        for tok in tokens:
            if tok == SOCIAL_EMOJI_TOKEN.lower():
                normalized_tokens.append(SOCIAL_EMOJI_TOKEN)
            elif tok.startswith(hashtag_prefix_lower):
                normalized_tokens.append(f"{SOCIAL_HASHTAG_PREFIX}{tok[len(hashtag_prefix_lower):]}")
            else:
                normalized_tokens.append(tok)
        tokens = normalized_tokens

    # Remove legacy control tokens from previous pipeline versions.
    tokens = [tok for tok in tokens if tok not in LEGACY_CONTROL_TOKENS]

    tokens_before_filter = list(tokens)

    # Apply minimum token length on token characters excluding underscores.
    if min_token_length > 1:
        tokens = [tok for tok in tokens if len(tok.replace("_", "")) >= min_token_length]
    if remove_stopwords:
        tokens = [tok for tok in tokens if tok.lower() not in stopwords]

    text_ml = " ".join(tokens)
    return text_ml, len(tokens_before_filter), len(tokens), tokens


# Detect likely pre-segmented text to avoid breaking existing underscore tokens.
def is_likely_presegmented(text: str) -> bool:
    tokens = [tok for tok in text.split() if tok]
    if not tokens:
        return False
    underscore_tokens = sum(1 for tok in tokens if "_" in tok)
    ratio = underscore_tokens / len(tokens)
    return underscore_tokens >= 2 and ratio >= 0.03


# Validate and parse ml_text config.
def parse_ml_config(config: dict[str, Any], repo_root: Path) -> dict[str, Any]:
    section = config.get("ml_text")
    if not isinstance(section, dict):
        raise PrepareMlTextError("configs/ml_text.yaml must define 'ml_text' mapping.")

    enabled = bool(section.get("enabled", True))
    input_files = section.get("input_files")
    if not isinstance(input_files, list) or not input_files:
        raise PrepareMlTextError("ml_text.input_files must be a non-empty list.")

    text_input_column = str(section.get("text_input_column", "")).strip()
    output_suffix = str(section.get("output_suffix", "_ml")).strip()
    if not text_input_column:
        raise PrepareMlTextError("ml_text.text_input_column must be non-empty.")
    if not output_suffix:
        raise PrepareMlTextError("ml_text.output_suffix must be non-empty.")

    min_token_length = int(section.get("min_token_length", 1))
    if min_token_length < 1:
        raise PrepareMlTextError("ml_text.min_token_length must be >= 1.")

    stopwords_path_value = str(section.get("stopwords_path", "")).strip()
    stopwords_path = (repo_root / stopwords_path_value).resolve() if stopwords_path_value else None

    segment_backend = str(section.get("segment_backend", "auto")).strip().lower()
    segment_model_dir_value = str(section.get("segment_model_dir", "")).strip()
    segment_model_dir = (repo_root / segment_model_dir_value).resolve() if segment_model_dir_value else None
    primary_variant = str(section.get("primary_variant", "lower")).strip().lower()
    if primary_variant not in {"seg", "lower"}:
        raise PrepareMlTextError("ml_text.primary_variant must be one of: seg, lower.")

    return {
        "enabled": enabled,
        "input_files": [str(p) for p in input_files],
        "text_input_column": text_input_column,
        "output_suffix": output_suffix,
        "replace_underscore_with_space": bool(section.get("replace_underscore_with_space", True)),
        "remove_url_tokens": bool(section.get("remove_url_tokens", True)),
        "keep_numbers": bool(section.get("keep_numbers", True)),
        "min_token_length": min_token_length,
        "remove_stopwords": bool(section.get("remove_stopwords", True)),
        "stopwords_path": stopwords_path,
        "log_top_tokens": int(section.get("log_top_tokens", 30)),
        "segment_backend": segment_backend,
        "segment_model_dir": segment_model_dir,
        "primary_variant": primary_variant,
        "preserve_social_emoji": bool(section.get("preserve_social_emoji", True)),
        "remove_news_emoji": bool(section.get("remove_news_emoji", True)),
        "preserve_social_punctuation_signals": bool(section.get("preserve_social_punctuation_signals", True)),
        "keep_sentence_boundary_token": bool(section.get("keep_sentence_boundary_token", True)),
        "respect_presegmented_input": bool(section.get("respect_presegmented_input", True)),
    }


# Build deterministic output path for one ML-prepared file.
def ml_output_path(input_path: Path, output_suffix: str) -> Path:
    return input_path.with_name(f"{input_path.stem}{output_suffix}{input_path.suffix}")


# Process one split CSV and generate a paired *_ml.csv file.
def process_one_file(
    input_rel_path: str,
    cfg: dict[str, Any],
    repo_root: Path,
    run_timestamp: str,
    stopwords: set[str],
    segment_backend_actual: str,
    segment_fn: Any,
    stage_warnings: list[str],
) -> tuple[dict[str, Any], list[dict[str, Any]], Counter, Counter]:
    input_path = (repo_root / input_rel_path).resolve()
    output_path = ml_output_path(input_path, cfg["output_suffix"])
    file_issues: list[dict[str, Any]] = []
    token_counter_seg: Counter = Counter()
    token_counter_lower: Counter = Counter()

    summary = {
        "run_timestamp": run_timestamp,
        "input_file": input_rel_path.replace("\\", "/"),
        "output_file": str(output_path.resolve().relative_to(repo_root.resolve()).as_posix()),
        "segment_backend_requested": cfg["segment_backend"],
        "segment_backend_actual": segment_backend_actual,
        "rows_input": 0,
        "rows_output": 0,
        "changed_count_seg": 0,
        "changed_count_seg_lower": 0,
        "changed_count": 0,
        "empty_text_ml_seg_count": 0,
        "empty_text_ml_seg_lower_count": 0,
        "empty_text_ml_count": 0,
        "empty_text_ml_ratio": 0.0,
        "avg_tokens_seg": 0.0,
        "avg_tokens_seg_lower": 0.0,
        "avg_tokens_before_filter": 0.0,
        "avg_tokens_after_filter": 0.0,
        "presegmented_passthrough_count": 0,
        "top_tokens_seg": "",
        "top_tokens_seg_lower": "",
        "top_tokens": "",
        "status": "PASS",
    }

    # Stop early when split file does not exist.
    if not input_path.exists():
        add_issue(
            file_issues,
            run_timestamp,
            summary["input_file"],
            "ERROR",
            "missing_input_file",
            1,
            f"expected_path={input_path.as_posix()}",
        )
        summary["status"] = "FAIL"
        return summary, file_issues, token_counter_seg, token_counter_lower

    df = pd.read_csv(input_path)
    summary["rows_input"] = int(len(df))

    # Enforce source text column existence before generating ML text.
    if cfg["text_input_column"] not in df.columns:
        add_issue(
            file_issues,
            run_timestamp,
            summary["input_file"],
            "ERROR",
            "missing_text_input_column",
            1,
            f"missing_column={cfg['text_input_column']}",
        )
        summary["status"] = "FAIL"
        return summary, file_issues, token_counter_seg, token_counter_lower

    text_series = df[cfg["text_input_column"]].fillna("").astype(str)
    content_type_series = (
        df["content_type"].fillna("news").astype(str).str.strip().str.lower()
        if "content_type" in df.columns
        else pd.Series(["news"] * len(df), index=df.index)
    )
    text_ml_seg_rows: list[str] = []
    text_ml_seg_lower_rows: list[str] = []
    before_counts: list[int] = []
    seg_counts: list[int] = []
    lower_counts: list[int] = []
    presegmented_passthrough_count = 0

    # Transform each row into segmented + lowercase-optional ML text variants.
    for text, content_type in zip(text_series.tolist(), content_type_series.tolist()):
        pre_cleaned = pre_clean_text(
            text=text,
            content_type=content_type,
            replace_underscore_with_space=cfg["replace_underscore_with_space"],
            remove_url_tokens=cfg["remove_url_tokens"],
            preserve_social_emoji=cfg["preserve_social_emoji"],
            remove_news_emoji=cfg["remove_news_emoji"],
            preserve_social_punctuation_signals=cfg["preserve_social_punctuation_signals"],
            keep_sentence_boundary_token=cfg["keep_sentence_boundary_token"],
        )
        if cfg["respect_presegmented_input"] and is_likely_presegmented(pre_cleaned):
            segmented_text = pre_cleaned
            presegmented_passthrough_count += 1
        else:
            segmented_text = segment_fn(pre_cleaned)
        text_ml_seg, tokens_before, tokens_after_seg, seg_tokens = tokenize_and_filter(
            text_segmented=segmented_text,
            lowercase=False,
            keep_numbers=cfg["keep_numbers"],
            min_token_length=cfg["min_token_length"],
            remove_stopwords=cfg["remove_stopwords"],
            stopwords=stopwords,
        )
        text_ml_seg_lower, _, tokens_after_lower, lower_tokens = tokenize_and_filter(
            text_segmented=segmented_text,
            lowercase=True,
            keep_numbers=cfg["keep_numbers"],
            min_token_length=cfg["min_token_length"],
            remove_stopwords=cfg["remove_stopwords"],
            stopwords=stopwords,
        )
        text_ml_seg_rows.append(text_ml_seg)
        text_ml_seg_lower_rows.append(text_ml_seg_lower)
        before_counts.append(tokens_before)
        seg_counts.append(tokens_after_seg)
        lower_counts.append(tokens_after_lower)
        token_counter_seg.update(seg_tokens)
        token_counter_lower.update(lower_tokens)

    out_df = df.copy()
    out_df["text_ml_seg"] = text_ml_seg_rows
    out_df["text_ml_seg_lower"] = text_ml_seg_lower_rows
    out_df["text_ml"] = out_df["text_ml_seg_lower"] if cfg["primary_variant"] == "lower" else out_df["text_ml_seg"]
    out_df["text_ml_seg_token_count"] = seg_counts
    out_df["text_ml_seg_lower_token_count"] = lower_counts
    out_df["text_ml_token_count"] = out_df["text_ml_seg_lower_token_count"] if cfg["primary_variant"] == "lower" else out_df["text_ml_seg_token_count"]
    out_df["flag_empty_text_ml_seg"] = out_df["text_ml_seg"] == ""
    out_df["flag_empty_text_ml_seg_lower"] = out_df["text_ml_seg_lower"] == ""
    out_df["flag_empty_text_ml"] = out_df["text_ml"] == ""
    summary["rows_output"] = int(len(out_df))

    # Keep row count unchanged because this stage only transforms text.
    if summary["rows_output"] != summary["rows_input"]:
        add_issue(
            file_issues,
            run_timestamp,
            summary["input_file"],
            "ERROR",
            "row_count_mismatch_after_prepare_ml_text",
            abs(summary["rows_output"] - summary["rows_input"]),
            f"rows_input={summary['rows_input']}, rows_output={summary['rows_output']}",
        )

    summary["changed_count_seg"] = int((text_series != out_df["text_ml_seg"]).sum())
    summary["changed_count_seg_lower"] = int((text_series != out_df["text_ml_seg_lower"]).sum())
    summary["changed_count"] = int((text_series != out_df["text_ml"]).sum())
    summary["empty_text_ml_seg_count"] = int(out_df["flag_empty_text_ml_seg"].sum())
    summary["empty_text_ml_seg_lower_count"] = int(out_df["flag_empty_text_ml_seg_lower"].sum())
    summary["empty_text_ml_count"] = int(out_df["flag_empty_text_ml"].sum())
    summary["empty_text_ml_ratio"] = (
        float(summary["empty_text_ml_count"] / summary["rows_output"])
        if summary["rows_output"] > 0
        else 0.0
    )
    summary["avg_tokens_before_filter"] = float(sum(before_counts) / len(before_counts)) if before_counts else 0.0
    summary["avg_tokens_seg"] = float(sum(seg_counts) / len(seg_counts)) if seg_counts else 0.0
    summary["avg_tokens_seg_lower"] = float(sum(lower_counts) / len(lower_counts)) if lower_counts else 0.0
    summary["avg_tokens_after_filter"] = summary["avg_tokens_seg_lower"] if cfg["primary_variant"] == "lower" else summary["avg_tokens_seg"]
    summary["presegmented_passthrough_count"] = int(presegmented_passthrough_count)
    top_n = max(1, int(cfg["log_top_tokens"]))
    summary["top_tokens_seg"] = " | ".join(
        [f"{token}:{count}" for token, count in token_counter_seg.most_common(top_n)]
    )
    summary["top_tokens_seg_lower"] = " | ".join(
        [f"{token}:{count}" for token, count in token_counter_lower.most_common(top_n)]
    )
    summary["top_tokens"] = summary["top_tokens_seg_lower"] if cfg["primary_variant"] == "lower" else summary["top_tokens_seg"]

    # Emit warning when generated ML text is unexpectedly empty at high ratio.
    if summary["empty_text_ml_ratio"] > EMPTY_RATIO_WARNING_THRESHOLD:
        add_issue(
            file_issues,
            run_timestamp,
            summary["input_file"],
            "WARNING",
            "high_empty_text_ml_ratio",
            summary["empty_text_ml_count"],
            f"ratio={summary['empty_text_ml_ratio']:.6f}, threshold={EMPTY_RATIO_WARNING_THRESHOLD:.2f}",
        )

    # Emit warning when stage had backend fallback warnings.
    fallback_messages = [msg for msg in stage_warnings if msg.startswith("segment_backend_fallback")]
    if fallback_messages:
        add_issue(
            file_issues,
            run_timestamp,
            summary["input_file"],
            "WARNING",
            "segment_backend_fallback",
            len(fallback_messages),
            " | ".join(fallback_messages),
        )

    summary["status"] = determine_status(file_issues)
    if summary["status"] == "FAIL":
        return summary, file_issues, token_counter_seg, token_counter_lower

    output_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(output_path, index=False, encoding="utf-8-sig")
    return summary, file_issues, token_counter_seg, token_counter_lower


# Write summary and issue logs for the prepare_ml_text stage.
def write_logs(repo_root: Path, summary_rows: list[dict[str, Any]], issue_rows: list[dict[str, Any]]) -> None:
    logs_dir = repo_root / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(summary_rows).to_csv(logs_dir / "prepare_ml_text_summary.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(issue_rows, columns=ISSUE_COLUMNS).to_csv(
        logs_dir / "prepare_ml_text_issues.csv", index=False, encoding="utf-8-sig"
    )


# Run ML-text preparation across configured split files.
def main() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    run_timestamp = datetime.now().astimezone().isoformat(timespec="seconds")
    config = load_yaml(repo_root / "configs" / "ml_text.yaml")
    cfg = parse_ml_config(config, repo_root)
    if not cfg["enabled"]:
        print("[PREPARE_ML_TEXT] Stage disabled by config (ml_text.enabled=false).")
        return

    stopwords: set[str] = set()
    if cfg["remove_stopwords"]:
        if cfg["stopwords_path"] is None:
            raise PrepareMlTextError("remove_stopwords=true requires ml_text.stopwords_path.")
        stopwords = load_stopwords(cfg["stopwords_path"])

    stage_warnings: list[str] = []
    segment_backend_actual, segment_fn = build_segmenter(cfg, stage_warnings)

    summary_rows: list[dict[str, Any]] = []
    issue_rows: list[dict[str, Any]] = []
    global_tokens_seg: Counter = Counter()
    global_tokens_lower: Counter = Counter()

    # Process each configured split file deterministically in declaration order.
    for input_rel_path in cfg["input_files"]:
        summary, file_issues, token_counter_seg, token_counter_lower = process_one_file(
            input_rel_path=input_rel_path,
            cfg=cfg,
            repo_root=repo_root,
            run_timestamp=run_timestamp,
            stopwords=stopwords,
            segment_backend_actual=segment_backend_actual,
            segment_fn=segment_fn,
            stage_warnings=stage_warnings,
        )
        summary_rows.append(summary)
        issue_rows.extend(file_issues)
        global_tokens_seg.update(token_counter_seg)
        global_tokens_lower.update(token_counter_lower)
        print(
            f"[PREPARE_ML_TEXT] input={summary['input_file']} status={summary['status']} "
            f"rows={summary['rows_output']} empty_ratio={summary['empty_text_ml_ratio']:.2%} "
            f"backend={segment_backend_actual}"
        )

    # Append one aggregated __ALL__ row to make reporting cross-check straightforward.
    total_input = int(sum(x["rows_input"] for x in summary_rows))
    total_output = int(sum(x["rows_output"] for x in summary_rows))
    total_changed_seg = int(sum(x["changed_count_seg"] for x in summary_rows))
    total_changed_seg_lower = int(sum(x["changed_count_seg_lower"] for x in summary_rows))
    total_changed = int(sum(x["changed_count"] for x in summary_rows))
    total_empty = int(sum(x["empty_text_ml_count"] for x in summary_rows))
    avg_before = float(sum(x["avg_tokens_before_filter"] * x["rows_output"] for x in summary_rows) / max(1, total_output))
    avg_seg = float(sum(x["avg_tokens_seg"] * x["rows_output"] for x in summary_rows) / max(1, total_output))
    avg_seg_lower = float(sum(x["avg_tokens_seg_lower"] * x["rows_output"] for x in summary_rows) / max(1, total_output))
    all_status = determine_status(issue_rows)
    all_row = {
        "run_timestamp": run_timestamp,
        "input_file": "__ALL__",
        "output_file": "__ALL__",
        "segment_backend_requested": cfg["segment_backend"],
        "segment_backend_actual": segment_backend_actual,
        "rows_input": total_input,
        "rows_output": total_output,
        "changed_count_seg": total_changed_seg,
        "changed_count_seg_lower": total_changed_seg_lower,
        "changed_count": total_changed,
        "empty_text_ml_seg_count": int(sum(x["empty_text_ml_seg_count"] for x in summary_rows)),
        "empty_text_ml_seg_lower_count": int(sum(x["empty_text_ml_seg_lower_count"] for x in summary_rows)),
        "empty_text_ml_count": total_empty,
        "empty_text_ml_ratio": (float(total_empty / total_output) if total_output > 0 else 0.0),
        "avg_tokens_before_filter": avg_before,
        "avg_tokens_seg": avg_seg,
        "avg_tokens_seg_lower": avg_seg_lower,
        "avg_tokens_after_filter": avg_seg_lower if cfg["primary_variant"] == "lower" else avg_seg,
        "presegmented_passthrough_count": int(sum(x.get("presegmented_passthrough_count", 0) for x in summary_rows)),
        "top_tokens_seg": " | ".join([f"{tok}:{cnt}" for tok, cnt in global_tokens_seg.most_common(cfg["log_top_tokens"])]),
        "top_tokens_seg_lower": " | ".join([f"{tok}:{cnt}" for tok, cnt in global_tokens_lower.most_common(cfg["log_top_tokens"])]),
        "top_tokens": " | ".join([f"{tok}:{cnt}" for tok, cnt in (global_tokens_lower if cfg["primary_variant"] == "lower" else global_tokens_seg).most_common(cfg["log_top_tokens"])]),
        "status": all_status,
    }
    summary_rows.append(all_row)

    write_logs(repo_root=repo_root, summary_rows=summary_rows, issue_rows=issue_rows)
    print(
        "[PREPARE_ML_TEXT] Generated logs/prepare_ml_text_summary.csv, "
        "logs/prepare_ml_text_issues.csv, and *_ml split files under data/processed."
    )

    if all_status == "FAIL":
        raise SystemExit("[PREPARE_ML_TEXT ERROR] Stage failed. Check logs/prepare_ml_text_issues.csv.")


# Expose a CLI-friendly entrypoint for local runs and CI checks.
if __name__ == "__main__":
    main()
