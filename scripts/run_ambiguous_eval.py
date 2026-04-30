import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
from groq import Groq
from sklearn.metrics import accuracy_score, confusion_matrix


OUTPUT_DIR = Path.home() / "resolv-experiments" / "experiment_results"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

AMBIGUOUS_KEYS = ["leakage", "seepage", "civil", "carpentry", "common area", "mason"]


def log(msg: str) -> None:
    print(msg, flush=True)


def norm(x) -> str:
    if pd.isna(x):
        return ""
    return str(x).strip()


def normalize_label(v: str) -> str:
    t = norm(v).lower()
    project_terms = ["project", "developer", "construction", "structural", "warranty", "waterproof", "dlp", "builder"]
    fm_terms = ["fm", "facility", "maintenance", "housekeeping", "security", "electrical", "plumbing", "operations"]
    if any(k in t for k in project_terms):
        return "Project"
    if any(k in t for k in fm_terms):
        return "FM"
    return "FM"


def find_data_file() -> Path:
    direct = Path.home() / "Downloads" / "GPL complaint data.xlsx"
    if direct.exists():
        return direct
    files = sorted((Path.home() / "Downloads").glob("*.xlsx"))
    if not files:
        raise FileNotFoundError("No .xlsx files in ~/Downloads")
    ranked = []
    for f in files:
        n = f.name.lower()
        score = sum(10 for k in ["gpl", "complaint", "tracker"] if k in n)
        ranked.append((score, f.stat().st_mtime, f))
    ranked.sort(reverse=True)
    return ranked[0][2]


def pick_sheet(path: Path) -> Tuple[str, pd.DataFrame]:
    xls = pd.ExcelFile(path)
    best = None
    best_score = -1
    for s in xls.sheet_names:
        d = pd.read_excel(path, sheet_name=s)
        cols = " ".join([str(c).lower() for c in d.columns])
        score = len(d)
        for k in ["complaint", "issue", "category", "flat", "site", "date"]:
            if k in cols:
                score += 100
        if score > best_score:
            best_score = score
            best = (s, d)
    return best


def choose_column(columns: List[str], keywords: List[str]) -> str:
    scored = []
    for c in columns:
        lc = c.lower()
        score = 0
        for i, k in enumerate(keywords):
            if k in lc:
                score += (len(keywords) - i) * 10
        scored.append((score, c))
    scored.sort(reverse=True)
    return scored[0][1]


def canonical_ambiguous(cat_text: str) -> str:
    c = re.sub(r"\s+", " ", norm(cat_text).lower())
    if "common" in c and "area" in c:
        return "common area"
    if "leak" in c:
        return "leakage"
    if "seep" in c:
        return "seepage"
    if "civil" in c:
        return "civil"
    if "carpen" in c:
        return "carpentry"
    if "mason" in c:
        return "mason"
    return ""


def cost_weighted_accuracy(y_true, y_pred):
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


def get_client() -> Groq:
    key = os.getenv("GROQ_API_KEY")
    if not key:
        env_file = Path("/Users/kartheek/resolv/.env")
        if env_file.exists():
            for line in env_file.read_text(encoding="utf-8").splitlines():
                if line.startswith("GROQ_API_KEY="):
                    key = line.split("=", 1)[1].strip().strip('"').strip("'")
                    break
    if not key:
        raise RuntimeError("GROQ_API_KEY not found in env or .env")
    return Groq(api_key=key)


def call_groq(client: Groq, prompt: str, retries: int = 4) -> str:
    for i in range(retries):
        try:
            resp = client.chat.completions.create(
                model="llama-3.1-70b-versatile",
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
            )
            return (resp.choices[0].message.content or "").strip()
        except Exception:
            if i == retries - 1:
                return ""
            time.sleep(1.5 * (i + 1))
    return ""


def parse_fm_project(response: str) -> str:
    u = response.strip().upper()
    if u.startswith("FM"):
        return "FM"
    if u.startswith("PROJECT"):
        return "Project"
    if "FM" in u:
        return "FM"
    if "PROJECT" in u:
        return "Project"
    return "FM"


def parse_confidence(response: str) -> float:
    m = re.search(r"CONFIDENCE\\s*:\\s*([0-9]*\\.?[0-9]+)", response.upper())
    if m:
        try:
            return float(m.group(1))
        except Exception:
            return 0.5
    nums = re.findall(r"([0-9]*\\.?[0-9]+)", response)
    if nums:
        try:
            return float(nums[0])
        except Exception:
            return 0.5
    return 0.5


def parse_prob(response: str) -> float:
    try:
        v = float(response.strip())
        return v if 0 <= v <= 1 else 0.5
    except Exception:
        nums = re.findall(r"([0-9]*\\.?[0-9]+)", response)
        if nums:
            try:
                v = float(nums[0])
                return v if 0 <= v <= 1 else 0.5
            except Exception:
                return 0.5
    return 0.5


def metrics(y_true: List[str], y_pred: List[str]) -> Dict:
    acc = float(accuracy_score(y_true, y_pred)) if y_true else 0.0
    cwa = float(cost_weighted_accuracy(y_true, y_pred)) if y_true else 0.0
    cm = confusion_matrix(y_true, y_pred, labels=["FM", "Project"]).tolist() if y_true else [[0, 0], [0, 0]]
    return {
        "accuracy_on_ambiguous": acc,
        "cost_weighted_accuracy": cwa,
        "sample_size": len(y_true),
        "confusion_matrix_labels": ["FM", "Project"],
        "confusion_matrix": cm,
    }


def main():
    data_file = find_data_file()
    sheet_name, raw_df = pick_sheet(data_file)
    raw_df.columns = [str(c).strip() for c in raw_df.columns]
    cols = raw_df.columns.tolist()

    complaint_col = choose_column(cols, ["description", "complaint", "issue", "text", "remarks", "title"])
    label_col = choose_column(cols, ["issue related", "ownership", "owner", "fm", "project", "type"])
    flat_col = choose_column(cols, ["flat", "unit", "apartment", "door"])
    site_col = choose_column(cols, ["site", "community", "society", "project"])
    category_col = choose_column(cols, ["sub category", "category", "trade", "type"])
    date_col = choose_column(cols, ["created date", "date", "logged", "opened"])

    # if both Category and Sub Category exist, prefer non-empty subcategory then category
    cat_col_lower = {c.lower(): c for c in cols}
    sub_col = cat_col_lower.get("sub category")
    main_col = cat_col_lower.get("category")

    df = pd.DataFrame(
        {
            "text": raw_df[complaint_col].astype(str),
            "raw_label": raw_df[label_col].astype(str),
            "flat_id": raw_df[flat_col].astype(str),
            "site_id": raw_df[site_col].astype(str),
            "category_fallback": raw_df[category_col].astype(str),
            "date": pd.to_datetime(raw_df[date_col], errors="coerce"),
        }
    )
    if sub_col is not None and main_col is not None:
        sub = raw_df[sub_col].astype(str)
        main = raw_df[main_col].astype(str)
        cat = sub.where(~sub.str.lower().isin(["nan", "", "none"]), main)
        df["category"] = cat
    else:
        df["category"] = df["category_fallback"]

    df["label"] = df["raw_label"].map(normalize_label)
    df = df[df["text"].str.strip() != ""].copy()
    df = df.dropna(subset=["date"]).copy()
    df["canonical_category"] = df["category"].map(canonical_ambiguous)

    # remove pre-possession communities
    df["month"] = df["date"].dt.to_period("M").astype(str)
    by_month = df.groupby(["site_id", "month"]).size().reset_index(name="cnt")
    avg_monthly = by_month.groupby("site_id")["cnt"].mean()
    mature_sites = set(avg_monthly[avg_monthly >= 10].index.tolist())
    mature_df = df[df["site_id"].isin(mature_sites)].copy()

    amb_df = mature_df[mature_df["canonical_category"] != ""].copy()

    # sample evenly across categories up to 200
    per_cat = {k: amb_df[amb_df["canonical_category"] == k] for k in ["leakage", "seepage", "civil", "carpentry", "common area", "mason"]}
    total_available = sum(len(v) for v in per_cat.values())
    if total_available <= 200:
        eval_df = amb_df.copy()
    else:
        cats_available = [k for k, v in per_cat.items() if len(v) > 0]
        base = 200 // len(cats_available)
        rem = 200 - base * len(cats_available)
        chunks = []
        for i, c in enumerate(cats_available):
            take = base + (1 if i < rem else 0)
            chunks.append(per_cat[c].sample(n=min(take, len(per_cat[c])), random_state=42))
        eval_df = pd.concat(chunks, ignore_index=False)
        # top-up if short due to sparse categories
        if len(eval_df) < 200:
            remaining = amb_df.drop(eval_df.index)
            add_n = min(200 - len(eval_df), len(remaining))
            if add_n > 0:
                eval_df = pd.concat([eval_df, remaining.sample(n=add_n, random_state=42)], ignore_index=False)

    eval_df = eval_df.sort_values("date").copy()
    eval_df = eval_df.drop_duplicates().copy()
    eval_df.to_csv(OUTPUT_DIR / "eval_ambiguous.csv", index=False)

    cat_counts = eval_df["canonical_category"].value_counts().to_dict()
    log(f"Loaded sheet: {sheet_name}")
    log("Category counts in eval_ambiguous.csv:")
    for k in ["leakage", "seepage", "civil", "carpentry", "common area", "mason"]:
        log(f"  {k}: {cat_counts.get(k, 0)}")

    # history source excludes eval rows
    history_df = mature_df.drop(index=eval_df.index, errors="ignore").sort_values("date").copy()
    client = get_client()

    # Groq direct baseline v3
    direct_rows = []
    direct_preds = []
    direct_true = []
    for _, r in eval_df.iterrows():
        prompt = (
            "You must respond with exactly one word only. Classify this Indian residential maintenance complaint as FM or Project. "
            "FM means facility management responsibility: plumbing, electrical, housekeeping, security, pest control, routine maintenance. "
            "Project means developer warranty responsibility: structural defects, waterproofing failures, construction quality issues, post-possession defects. "
            f"Category: {r['category']}. Complaint: {r['text']}. Respond with one word only: FM or Project."
        )
        raw_resp = call_groq(client, prompt)
        pred = parse_fm_project(raw_resp)
        direct_rows.append(
            {
                "row_index": int(r.name),
                "category": r["category"],
                "canonical_category": r["canonical_category"],
                "text": r["text"],
                "true_label": r["label"],
                "prediction": pred,
                "raw_response": raw_resp,
            }
        )
        direct_preds.append(pred)
        direct_true.append(r["label"])

    m_direct = metrics(direct_true, direct_preds)
    m_direct["overall_accuracy"] = m_direct["accuracy_on_ambiguous"]
    m_direct["ambiguous_category_accuracy"] = m_direct["accuracy_on_ambiguous"]
    m_direct["rows"] = direct_rows
    (OUTPUT_DIR / "results_groq_direct_v3.json").write_text(json.dumps(m_direct, indent=2), encoding="utf-8")

    log("First 5 raw Groq direct responses:")
    for x in direct_rows[:5]:
        log(f"- {x['raw_response']}")

    # ARIA Tier 2 on same set
    tier2_rows = []
    tier2_true = []
    tier2_pred = []
    for _, r in eval_df.iterrows():
        hist = history_df[history_df["flat_id"] == r["flat_id"]].tail(3)
        prior = "None" if len(hist) == 0 else " | ".join([f"{h['category']}: {h['text']}" for _, h in hist.iterrows()])
        prompt = (
            "You are an intelligent complaint routing system for a residential gated community in India. "
            "Determine whether this complaint is FM or Project responsibility using complaint details and prior flat history. "
            f"Category: {r['category']}. Complaint: {r['text']}. Prior complaints from this flat: {prior}. "
            "Respond exactly in this format: Decision: FM or Project. Confidence: [0 to 1]. Reasoning: one sentence."
        )
        raw_resp = call_groq(client, prompt)
        pred = parse_fm_project(raw_resp)
        conf = parse_confidence(raw_resp)
        tier2_rows.append(
            {
                "row_index": int(r.name),
                "category": r["category"],
                "canonical_category": r["canonical_category"],
                "text": r["text"],
                "true_label": r["label"],
                "prediction": pred,
                "confidence": conf,
                "escalated": conf < 0.65,
                "prior": prior,
                "raw_response": raw_resp,
            }
        )
        tier2_true.append(r["label"])
        tier2_pred.append(pred)

    m_tier2 = metrics(tier2_true, tier2_pred)
    m_tier2["overall_accuracy"] = m_tier2["accuracy_on_ambiguous"]
    m_tier2["ambiguous_category_accuracy"] = m_tier2["accuracy_on_ambiguous"]
    m_tier2["escalated_count"] = int(sum(1 for x in tier2_rows if x["escalated"]))
    m_tier2["rows"] = tier2_rows
    (OUTPUT_DIR / "results_aria_tier2_ambiguous.json").write_text(json.dumps(m_tier2, indent=2), encoding="utf-8")

    # ARIA Tier 3 on same set
    tier3_rows = []
    tier3_true = []
    tier3_pred = []

    def run_three_prompts(cat: str, text: str, prior: str):
        p1 = (
            "You are analyzing a maintenance complaint to assess if it is a plumbing or pipe failure. "
            "This would be FM responsibility. Assess the probability between 0 and 1. "
            f"Category: {cat}. Complaint: {text}. Prior history: {prior}. "
            "Respond with only a decimal number between 0 and 1. Nothing else."
        )
        p2 = (
            "You are analyzing a maintenance complaint to assess if it is a structural defect, waterproofing failure, "
            "or construction quality issue. This would be developer Project responsibility under warranty. Assess the probability between 0 and 1. "
            f"Category: {cat}. Complaint: {text}. Prior history: {prior}. "
            "Respond with only a decimal number between 0 and 1. Nothing else."
        )
        p3 = (
            "You are analyzing a maintenance complaint to assess if it is an asset or equipment failure. "
            "This would be FM responsibility. Assess the probability between 0 and 1. "
            f"Category: {cat}. Complaint: {text}. Prior history: {prior}. "
            "Respond with only a decimal number between 0 and 1. Nothing else."
        )
        with ThreadPoolExecutor(max_workers=3) as ex:
            futs = [ex.submit(call_groq, client, p) for p in [p1, p2, p3]]
            r1, r2, r3 = [f.result() for f in futs]
        return r1, r2, r3

    for _, r in eval_df.iterrows():
        hist = history_df[history_df["flat_id"] == r["flat_id"]].tail(3)
        prior = "None" if len(hist) == 0 else " | ".join([f"{h['category']}: {h['text']}" for _, h in hist.iterrows()])
        rp, rs, ra = run_three_prompts(r["category"], r["text"], prior)
        pp, ps, pa = parse_prob(rp), parse_prob(rs), parse_prob(ra)

        score_pipe = pp * 1.0
        score_struct = ps * 10.0
        score_asset = pa * 3.0
        pred = "Project" if score_struct >= max(score_pipe, score_asset) else "FM"

        tier3_rows.append(
            {
                "row_index": int(r.name),
                "category": r["category"],
                "canonical_category": r["canonical_category"],
                "text": r["text"],
                "true_label": r["label"],
                "prediction": pred,
                "pipe_probability": pp,
                "structural_probability": ps,
                "asset_probability": pa,
            }
        )
        tier3_true.append(r["label"])
        tier3_pred.append(pred)

    m_tier3 = metrics(tier3_true, tier3_pred)
    m_tier3["overall_accuracy"] = m_tier3["accuracy_on_ambiguous"]
    m_tier3["ambiguous_category_accuracy"] = m_tier3["accuracy_on_ambiguous"]
    m_tier3["rows"] = tier3_rows
    (OUTPUT_DIR / "results_aria_tier3_ambiguous.json").write_text(json.dumps(m_tier3, indent=2), encoding="utf-8")

    # final comparison table
    lines = [
        "System | Accuracy on Ambiguous | Cost-Weighted Accuracy | Sample Size",
        f"Groq direct no context | {m_direct['accuracy_on_ambiguous']:.4f} | {m_direct['cost_weighted_accuracy']:.4f} | {m_direct['sample_size']}",
        f"ARIA Tier 2 with context | {m_tier2['accuracy_on_ambiguous']:.4f} | {m_tier2['cost_weighted_accuracy']:.4f} | {m_tier2['sample_size']}",
        f"ARIA Tier 3 cost-of-error | {m_tier3['accuracy_on_ambiguous']:.4f} | {m_tier3['cost_weighted_accuracy']:.4f} | {m_tier3['sample_size']}",
    ]
    txt = "\n".join(lines)
    (OUTPUT_DIR / "ambiguous_comparison_final.txt").write_text(txt, encoding="utf-8")

    log("\nFinal comparison table (ambiguous only):")
    log(txt)


if __name__ == "__main__":
    main()
