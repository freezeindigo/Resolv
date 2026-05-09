"""
Build eval_ambiguous.csv from GPL complaint Excel, then run Groq direct v3,
ARIA Tier 2, and ARIA Tier 3 on that set with identical complaints.
Prior history for Tier 2/3: full mature dataset minus eval_ambiguous rows.
"""
import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
from groq import Groq
from sklearn.metrics import accuracy_score, confusion_matrix

# Reuse helpers aligned with run_full_experiment.py
OUTPUT_DIR = Path.home() / "resolv-experiments" / "experiment_results"
DATA_FILE = Path.home() / "Downloads" / "GPL complaint data.xlsx"
GROQ_MODEL = "llama-3.1-70b-versatile"

# Match user wording: six buckets for filtering + even sampling
BUCKET_DEFS: List[Tuple[str, List[str]]] = [
    ("leakage", ["leakage"]),
    ("seepage", ["seepage"]),
    ("civil", ["civil"]),
    ("carpentry", ["carpentry", "carpent"]),  # typo "Carpentary"
    ("common area", ["common area"]),
    ("mason", ["mason"]),
]


def resolve_data_file() -> Path:
    if DATA_FILE.exists():
        return DATA_FILE
    downloads = Path.home() / "Downloads"
    candidates = sorted(downloads.glob("*.xlsx"))
    if not candidates:
        raise FileNotFoundError(f"No .xlsx files found in {downloads}")
    ranked = []
    for p in candidates:
        name = p.name.lower()
        score = 0
        for k in ["gpl", "complaint", "tracker", "resolv"]:
            if k in name:
                score += 10
        ranked.append((score, p.stat().st_mtime, p))
    ranked.sort(reverse=True)
    return ranked[0][2]


def norm_text(x) -> str:
    if pd.isna(x):
        return ""
    return str(x).strip()


def clean_category(x) -> str:
    return re.sub(r"\s+", " ", norm_text(x).lower())


def pick_sheet(excel_path: Path) -> Tuple[str, pd.DataFrame]:
    xls = pd.ExcelFile(excel_path)
    best_name = None
    best_df = None
    best_score = -1
    for sheet in xls.sheet_names:
        df = pd.read_excel(excel_path, sheet_name=sheet)
        cols = [c.lower() for c in df.columns.astype(str)]
        score = len(df)
        for k in ["complaint", "description", "issue", "remarks", "category", "flat", "unit", "site", "community", "date"]:
            if any(k in c for c in cols):
                score += 200
        if score > best_score:
            best_score = score
            best_name = sheet
            best_df = df
    return best_name, best_df


def choose_column(columns: List[str], keywords: List[str]) -> str:
    scores = []
    for col in columns:
        lc = col.lower()
        score = sum((len(keywords) - i) * 10 for i, k in enumerate(keywords) if k in lc)
        scores.append((score, col))
    scores.sort(reverse=True)
    return scores[0][1]


def map_labels(raw_label: str) -> str:
    t = norm_text(raw_label).lower()
    if t == "":
        return "FM"
    project_terms = [
        "project", "developer", "construction", "structural", "warranty",
        "waterproof", "dlp", "builder",
    ]
    fm_terms = [
        "fm", "facility", "maintenance", "housekeeping", "security",
        "electrical", "plumbing", "operations",
    ]
    if any(k in t for k in project_terms):
        return "Project"
    if any(k in t for k in fm_terms):
        return "FM"
    if t in {"owner", "owners"}:
        return "Project"
    return "FM"


def cost_weighted_accuracy(y_true: List[str], y_pred: List[str]) -> float:
    total_cost = 0
    max_cost = 0
    for true, pred in zip(y_true, y_pred):
        if true == "Project" and pred == "FM":
            total_cost += 10
            max_cost += 10
        elif true == "FM" and pred == "Project":
            total_cost += 1
            max_cost += 10
        else:
            max_cost += 10
    return 1 - (total_cost / max_cost) if max_cost > 0 else 1.0


def compute_metrics_ambiguous(y_true: List[str], y_pred: List[str]) -> Dict:
    n = len(y_true)
    overall = float(accuracy_score(y_true, y_pred)) if n else 0.0
    cwa = float(cost_weighted_accuracy(y_true, y_pred)) if n else 0.0
    cm = confusion_matrix(y_true, y_pred, labels=["FM", "Project"]).tolist() if n else [[0, 0], [0, 0]]
    return {
        "overall_accuracy": overall,
        "ambiguous_category_accuracy": overall,
        "cost_weighted_accuracy": cwa,
        "confusion_matrix_labels": ["FM", "Project"],
        "confusion_matrix": cm,
        "sample_size": n,
    }


def assign_bucket(cat_clean: str) -> Optional[str]:
    for bucket_name, keys in BUCKET_DEFS:
        for k in keys:
            if k in cat_clean:
                return bucket_name
    return None


def parse_groq_one_word_v3(text: str) -> str:
    t = norm_text(text).upper()
    if t.startswith("PROJECT"):
        return "Project"
    if t.startswith("FM"):
        return "FM"
    if "PROJECT" in t:
        return "Project"
    if re.search(r"\bFM\b", t) or " FM" in t or t == "FM":
        return "FM"
    if "FM" in t:
        return "FM"
    return "FM"


def parse_decision_tier2(text: str) -> str:
    low = norm_text(text).lower()
    if "project" in low:
        return "Project"
    if "fm" in low:
        return "FM"
    return "FM"


def parse_confidence(text: str) -> float:
    m = re.search(r"confidence\s*:\s*([0-9]*\.?[0-9]+)", norm_text(text).lower())
    if m:
        try:
            return max(0.0, min(1.0, float(m.group(1))))
        except ValueError:
            return 0.5
    nums = re.findall(r"([0-9]*\.?[0-9]+)", norm_text(text))
    if nums:
        try:
            return max(0.0, min(1.0, float(nums[0])))
        except ValueError:
            return 0.5
    return 0.5


def parse_prob(text: str) -> float:
    try:
        v = float(norm_text(text))
        return v if 0 <= v <= 1 else 0.5
    except ValueError:
        m = re.search(r"([0-9]*\.?[0-9]+)", norm_text(text))
        if m:
            try:
                v = float(m.group(1))
                if 0 <= v <= 1:
                    return v
            except ValueError:
                pass
    return 0.5


def call_groq(client: Groq, prompt: str) -> str:
    for i in range(6):
        try:
            res = client.chat.completions.create(
                model=GROQ_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
            )
            return (res.choices[0].message.content or "").strip()
        except Exception:
            if i == 5:
                return ""
            time.sleep(1.2 * (i + 1))
    return ""


def load_mature_frame(excel_path: Path) -> pd.DataFrame:
    _, raw_df = pick_sheet(excel_path)
    raw_df.columns = [str(c).strip() for c in raw_df.columns]
    cols = raw_df.columns.tolist()

    complaint_col = choose_column(cols, ["complaint title", "description", "complaint", "issue", "text", "remarks"])
    label_col = choose_column(cols, ["issue related", "ownership", "owner", "fm", "project", "category", "type"])
    flat_col = choose_column(cols, ["flat", "unit", "apartment", "door", "house"])
    comm_col = choose_column(cols, ["site name", "community", "site", "society", "project"])
    # Prefer main Category column for trade (not Sub Category)
    if "Category" in raw_df.columns:
        cat_col = "Category"
    else:
        cat_col = choose_column(cols, ["category", "sub category", "trade", "type"])
    date_col = choose_column(cols, ["created date", "date", "logged", "opened", "timestamp"])

    ticket_col = None
    for c in cols:
        if "ticket" in c.lower() and "id" in c.lower():
            ticket_col = c
            break

    df = pd.DataFrame(
        {
            "text": raw_df[complaint_col].astype(str).fillna(""),
            "raw_label": raw_df[label_col].astype(str).fillna(""),
            "flat_id": raw_df[flat_col].astype(str).fillna(""),
            "community_id": raw_df[comm_col].astype(str).fillna(""),
            "category": raw_df[cat_col].astype(str).fillna(""),
            "date_raw": raw_df[date_col],
        }
    )
    if ticket_col:
        df["ticket_id"] = raw_df[ticket_col].astype(str).fillna("")
    else:
        df["ticket_id"] = ""

    df["date"] = pd.to_datetime(df["date_raw"], errors="coerce")
    df["label"] = df["raw_label"].map(map_labels)
    df = df[df["text"].str.strip() != ""].copy()

    # Mature communities first (avg complaints per month >= 10), then ambiguous filter
    if df["date"].notna().any():
        tmp = df.dropna(subset=["date"]).copy()
        tmp["month"] = tmp["date"].dt.to_period("M").astype(str)
        by_month = tmp.groupby(["community_id", "month"]).size().reset_index(name="cnt")
        avg_monthly = by_month.groupby("community_id")["cnt"].mean()
        mature = avg_monthly[avg_monthly >= 10].index
        df = df[df["community_id"].isin(mature)].copy()

    df["category_clean"] = df["category"].map(clean_category)
    df["ambiguity_bucket"] = df["category_clean"].map(assign_bucket)
    df = df[df["ambiguity_bucket"].notna()].copy()

    if len(df) < 50:
        # fallback like full experiment
        _, raw_df2 = pick_sheet(excel_path)
        raw_df2.columns = [str(c).strip() for c in raw_df2.columns]
        cols2 = raw_df2.columns.tolist()
        complaint_col = choose_column(cols2, ["complaint title", "description", "complaint", "issue", "text", "remarks"])
        label_col = choose_column(cols2, ["issue related", "ownership", "owner", "fm", "project", "category", "type"])
        flat_col = choose_column(cols2, ["flat", "unit", "apartment", "door", "house"])
        comm_col = choose_column(cols2, ["site name", "community", "site", "society", "project"])
        cat_col = "Category" if "Category" in raw_df2.columns else choose_column(cols2, ["category", "sub category", "trade", "type"])
        date_col = choose_column(cols2, ["created date", "date", "logged", "opened", "timestamp"])
        ticket_col = None
        for c in cols2:
            if "ticket" in c.lower() and "id" in c.lower():
                ticket_col = c
                break
        df = pd.DataFrame(
            {
                "text": raw_df2[complaint_col].astype(str).fillna(""),
                "raw_label": raw_df2[label_col].astype(str).fillna(""),
                "flat_id": raw_df2[flat_col].astype(str).fillna(""),
                "community_id": raw_df2[comm_col].astype(str).fillna(""),
                "category": raw_df2[cat_col].astype(str).fillna(""),
                "date_raw": raw_df2[date_col],
            }
        )
        if ticket_col:
            df["ticket_id"] = raw_df2[ticket_col].astype(str).fillna("")
        else:
            df["ticket_id"] = ""
        df["date"] = pd.to_datetime(df["date_raw"], errors="coerce")
        df["label"] = df["raw_label"].map(map_labels)
        df = df[df["text"].str.strip() != ""].copy()
        if df["date"].notna().any():
            tmp = df.dropna(subset=["date"]).copy()
            tmp["month"] = tmp["date"].dt.to_period("M").astype(str)
            by_month = tmp.groupby(["community_id", "month"]).size().reset_index(name="cnt")
            avg_monthly = by_month.groupby("community_id")["cnt"].mean()
            mature = avg_monthly[avg_monthly >= 10].index
            df = df[df["community_id"].isin(mature)].copy()
        df["category_clean"] = df["category"].map(clean_category)
        df["ambiguity_bucket"] = df["category_clean"].map(assign_bucket)
        df = df[df["ambiguity_bucket"].notna()].copy()

    return df


def row_key(r: pd.Series) -> str:
    tid = norm_text(r.get("ticket_id", ""))
    if tid:
        return f"tid:{tid}"
    return f"h:{hash(norm_text(r['text']))}|f:{norm_text(r['flat_id'])}|d:{r.get('date', '')}"


def sample_even_per_bucket(df: pd.DataFrame, max_total: int = 200) -> pd.DataFrame:
    buckets = [b for b, _ in BUCKET_DEFS]
    parts = []
    for b in buckets:
        sub = df[df["ambiguity_bucket"] == b]
        if len(sub):
            parts.append(sub)
    if not parts:
        return df.iloc[0:0]
    combined = pd.concat(parts, ignore_index=True)
    n = len(combined)
    if n <= max_total:
        return combined.sample(frac=1.0, random_state=42).reset_index(drop=True)
    per = max_total // len(buckets)
    rem = max_total - per * len(buckets)
    out_chunks = []
    extra_buckets = buckets[:rem] if rem else []
    for b in buckets:
        sub = combined[combined["ambiguity_bucket"] == b]
        take = per + (1 if b in extra_buckets else 0)
        take = min(take, len(sub))
        if take > 0:
            out_chunks.append(sub.sample(n=take, random_state=42))
    out = pd.concat(out_chunks, ignore_index=True)
    if len(out) < max_total:
        used = set(out.apply(row_key, axis=1))
        rest = combined[~combined.apply(lambda r: row_key(r) in used, axis=1)]
        need = max_total - len(out)
        if len(rest) >= need:
            out = pd.concat([out, rest.sample(n=need, random_state=43)], ignore_index=True)
    return out.sample(frac=1.0, random_state=44).reset_index(drop=True)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    excel_path = resolve_data_file()
    print(f"Using data file: {excel_path}", flush=True)

    full_mature = load_mature_frame(excel_path)
    print(f"Mature-community ambiguous-category pool size: {len(full_mature)}", flush=True)

    counts_before = full_mature.groupby("ambiguity_bucket").size().reindex([b for b, _ in BUCKET_DEFS], fill_value=0)
    print("Counts per category (pool, before cap):", flush=True)
    for k, v in counts_before.items():
        print(f"  {k}: {int(v)}", flush=True)

    eval_amb = sample_even_per_bucket(full_mature, max_total=200)
    eval_keys = set(eval_amb.apply(row_key, axis=1))
    hist_source = full_mature[~full_mature.apply(lambda r: row_key(r) in eval_keys, axis=1)].copy()
    if hist_source["date"].notna().any():
        hist_source = hist_source.sort_values("date")

    out_csv = OUTPUT_DIR / "eval_ambiguous.csv"
    eval_amb.to_csv(out_csv, index=False)
    print(f"Saved {out_csv} (n={len(eval_amb)})", flush=True)
    counts_eval = eval_amb.groupby("ambiguity_bucket").size().reindex([b for b, _ in BUCKET_DEFS], fill_value=0)
    print("Counts per category (eval_ambiguous.csv):", flush=True)
    for k, v in counts_eval.items():
        print(f"  {k}: {int(v)}", flush=True)

    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        env_path = Path(__file__).parent.parent / ".env"
        if env_path.exists():
            for line in env_path.read_text().splitlines():
                if line.startswith("GROQ_API_KEY="):
                    api_key = line.split("=", 1)[1].strip().strip('"').strip("'")
                    break
    if not api_key:
        raise RuntimeError("GROQ_API_KEY not set")
    client = Groq(api_key=api_key)

    # --- Groq direct v3 ---
    y_true = eval_amb["label"].astype(str).tolist()
    y_pred_v3: List[str] = []
    raw_samples: List[str] = []
    for i, (_, r) in enumerate(eval_amb.iterrows()):
        prompt = (
            "You must respond with exactly one word only. Classify this Indian residential maintenance complaint as FM or Project. "
            "FM means facility management responsibility: plumbing, electrical, housekeeping, security, pest control, routine maintenance. "
            "Project means developer warranty responsibility: structural defects, waterproofing failures, construction quality issues, post-possession defects. "
            f"Category: {r['category']}. Complaint: {r['text']}. Respond with one word only: FM or Project."
        )
        raw = call_groq(client, prompt)
        y_pred_v3.append(parse_groq_one_word_v3(raw))
        if i < 5:
            raw_samples.append(raw)

    m_v3 = compute_metrics_ambiguous(y_true, y_pred_v3)
    m_v3["predictions"] = y_pred_v3
    m_v3["sample_raw_responses_first5"] = raw_samples
    v3_path = OUTPUT_DIR / "results_groq_direct_v3.json"
    with open(v3_path, "w", encoding="utf-8") as f:
        json.dump(m_v3, f, indent=2)
    print("Groq direct v3:", json.dumps({k: m_v3[k] for k in ["overall_accuracy", "ambiguous_category_accuracy", "cost_weighted_accuracy", "sample_size"]}, indent=2), flush=True)
    print("First 5 raw Groq responses:", flush=True)
    for j, s in enumerate(raw_samples):
        print(f"  [{j+1}] {repr(s)}", flush=True)

    # --- ARIA Tier 2 (full eval set; metrics on all rows = ambiguous) ---
    tier2_preds = []
    eval_reset = eval_amb.reset_index(drop=True)
    for idx, r in eval_reset.iterrows():
        hist = hist_source[hist_source["flat_id"].astype(str) == str(r["flat_id"])].tail(3)
        if len(hist) == 0:
            prior = "None"
        else:
            prior = " | ".join([f"{h['category']}: {h['text']}" for _, h in hist.iterrows()])
        prompt = (
            "You are an intelligent complaint routing system for a residential gated community in India. "
            "Your job is to determine whether this complaint is the responsibility of the FM company (routine maintenance) "
            "or the Project team (developer warranty and structural issues). Use the complaint details and prior history from this flat "
            f"to make your decision. Category: {r['category']}. Complaint: {r['text']}. Prior complaints from this flat: {prior}. "
            "Think step by step about what could cause this issue and who is responsible. Respond in exactly this format: "
            "Decision: FM or Project. Confidence: [number between 0 and 1]. Reasoning: one sentence."
        )
        resp = call_groq(client, prompt)
        decision = parse_decision_tier2(resp)
        conf = parse_confidence(resp)
        tier2_preds.append(
            {
                "index": int(idx),
                "category": str(r["category"]),
                "text": str(r["text"]),
                "true_label": str(r["label"]),
                "decision": decision,
                "confidence": float(conf),
                "escalated": bool(conf < 0.65),
                "prior": prior,
            }
        )

    y_pred_t2 = [p["decision"] for p in tier2_preds]
    m_t2 = compute_metrics_ambiguous(y_true, y_pred_t2)
    m_t2["predictions"] = tier2_preds
    m_t2["escalated_count"] = int(sum(1 for p in tier2_preds if p["escalated"]))
    t2_path = OUTPUT_DIR / "results_aria_tier2_ambiguous.json"
    with open(t2_path, "w", encoding="utf-8") as f:
        json.dump(m_t2, f, indent=2)
    print("ARIA Tier 2 ambiguous:", json.dumps({k: m_t2[k] for k in ["overall_accuracy", "ambiguous_category_accuracy", "cost_weighted_accuracy", "sample_size", "escalated_count"]}, indent=2), flush=True)

    # --- ARIA Tier 3 (all rows in eval_ambiguous) ---
    prior_map = {p["index"]: p["prior"] for p in tier2_preds}
    tier3_rows = []
    changed = 0
    changed_ok = 0
    for idx, r in eval_reset.iterrows():
        prior = prior_map.get(int(idx), "None")
        p1 = (
            "You are analyzing a maintenance complaint to assess if it is a plumbing or pipe failure. "
            "This would be FM responsibility. Assess the probability between 0 and 1. "
            f"Category: {r['category']}. Complaint: {r['text']}. Prior history: {prior}. "
            "Respond with only a decimal number between 0 and 1. Nothing else."
        )
        p2 = (
            "You are analyzing a maintenance complaint to assess if it is a structural defect, waterproofing failure, "
            "or construction quality issue. This would be developer Project responsibility under warranty. "
            "Assess the probability between 0 and 1. "
            f"Category: {r['category']}. Complaint: {r['text']}. Prior history: {prior}. "
            "Respond with only a decimal number between 0 and 1. Nothing else."
        )
        p3 = (
            "You are analyzing a maintenance complaint to assess if it is an asset or equipment failure. "
            "This would be FM responsibility. Assess the probability between 0 and 1. "
            f"Category: {r['category']}. Complaint: {r['text']}. Prior history: {prior}. "
            "Respond with only a decimal number between 0 and 1. Nothing else."
        )
        with ThreadPoolExecutor(max_workers=3) as ex:
            futs = [ex.submit(call_groq, client, p) for p in [p1, p2, p3]]
            resp_pipe, resp_struct, resp_asset = [f.result() for f in futs]
        prob_pipe = parse_prob(resp_pipe)
        prob_struct = parse_prob(resp_struct)
        prob_asset = parse_prob(resp_asset)
        score_pipe = prob_pipe * 1.0
        score_struct = prob_struct * 10.0
        score_asset = prob_asset * 3.0
        weighted_dec = "Project" if score_struct >= max(score_pipe, score_asset) else "FM"
        raw_max = max(prob_pipe, prob_struct, prob_asset)
        pure_dec = "Project" if raw_max == prob_struct else "FM"
        if weighted_dec != pure_dec:
            changed += 1
            if weighted_dec == r["label"]:
                changed_ok += 1
        tier3_rows.append(
            {
                "index": int(idx),
                "weighted_decision": weighted_dec,
                "pure_probability_decision": pure_dec,
                "pipe_probability": prob_pipe,
                "structural_probability": prob_struct,
                "asset_probability": prob_asset,
            }
        )

    y_pred_t3 = [x["weighted_decision"] for x in tier3_rows]
    m_t3 = compute_metrics_ambiguous(y_true, y_pred_t3)
    m_t3["predictions"] = tier3_rows
    m_t3["cost_arbiter_changed_decisions"] = changed
    m_t3["changed_decisions_matching_ground_truth"] = changed_ok
    t3_path = OUTPUT_DIR / "results_aria_tier3_ambiguous.json"
    with open(t3_path, "w", encoding="utf-8") as f:
        json.dump(m_t3, f, indent=2)
    print("ARIA Tier 3 ambiguous:", json.dumps({k: m_t3[k] for k in ["overall_accuracy", "ambiguous_category_accuracy", "cost_weighted_accuracy", "sample_size", "cost_arbiter_changed_decisions"]}, indent=2), flush=True)

    # Comparison table
    lines = [
        "System | Accuracy on Ambiguous | Cost-Weighted Accuracy | Sample Size",
        f"Groq direct no context | {m_v3['ambiguous_category_accuracy']:.6f} | {m_v3['cost_weighted_accuracy']:.6f} | {m_v3['sample_size']}",
        f"ARIA Tier 2 with context | {m_t2['ambiguous_category_accuracy']:.6f} | {m_t2['cost_weighted_accuracy']:.6f} | {m_t2['sample_size']}",
        f"ARIA Tier 3 cost-of-error | {m_t3['ambiguous_category_accuracy']:.6f} | {m_t3['cost_weighted_accuracy']:.6f} | {m_t3['sample_size']}",
    ]
    table_txt = "\n".join(lines)
    print("\n" + table_txt + "\n", flush=True)
    cmp_path = OUTPUT_DIR / "ambiguous_comparison_final.txt"
    cmp_path.write_text(table_txt + "\n", encoding="utf-8")
    print(f"Wrote {cmp_path}", flush=True)


if __name__ == "__main__":
    main()
