import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeout
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
from groq import Groq
from sklearn.metrics import accuracy_score


OUT_DIR = Path.home() / "resolv-experiments" / "experiment_results"
DEFAULT_DATA = Path.home() / "Downloads" / "GPL complaint data.xlsx"
GROQ_MODEL = "llama-3.1-70b-versatile"


def log(msg: str) -> None:
    print(msg, flush=True)


def norm(x) -> str:
    if pd.isna(x):
        return ""
    return str(x).strip()


def clean(x) -> str:
    return re.sub(r"\s+", " ", norm(x).lower())


def resolve_data_file() -> Path:
    if DEFAULT_DATA.exists():
        return DEFAULT_DATA
    downloads = Path.home() / "Downloads"
    candidates = sorted(downloads.glob("*.xlsx"))
    if not candidates:
        raise FileNotFoundError(f"No .xlsx files found in {downloads}")
    ranked = []
    for p in candidates:
        name = p.name.lower()
        score = 0
        for k in ["gpl", "complaint", "tracker"]:
            if k in name:
                score += 10
        ranked.append((score, p.stat().st_mtime, p))
    ranked.sort(reverse=True)
    return ranked[0][2]


def pick_sheet(path: Path):
    xls = pd.ExcelFile(path)
    best_name = None
    best_df = None
    best_score = -1
    for s in xls.sheet_names:
        df = pd.read_excel(path, sheet_name=s)
        cols = [str(c).lower() for c in df.columns]
        score = len(df)
        for k in ["complaint", "title", "category", "flat", "site", "date", "project", "fm"]:
            if any(k in c for c in cols):
                score += 200
        if score > best_score:
            best_score = score
            best_name = s
            best_df = df
    return best_name, best_df


def map_label(v: str) -> str:
    t = clean(v)
    if "project" in t or "developer" in t or "warranty" in t or "struct" in t or "dlp" in t:
        return "Project"
    if "fm" in t or "facility" in t or "maint" in t or "plumb" in t or "elect" in t or "security" in t:
        return "FM"
    if t == "owner":
        return "Project"
    return "FM"


def map_ambiguous_category(c: str) -> Optional[str]:
    t = clean(c)
    if "leakage" in t:
        return "leakage"
    if "seepage" in t:
        return "seepage"
    if "civil" in t:
        return "civil"
    if "carpent" in t:
        return "carpentry"
    if "common area" in t or ("common" in t and "area" in t):
        return "common area"
    if "mason" in t:
        return "mason"
    return None


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


def get_client() -> Groq:
    key = os.getenv("GROQ_API_KEY")
    if not key:
        env_path = Path(__file__).parent.parent / ".env"
        if env_path.exists():
            for line in env_path.read_text(encoding="utf-8").splitlines():
                if line.startswith("GROQ_API_KEY="):
                    key = line.split("=", 1)[1].strip().strip('"').strip("'")
                    break
    if not key:
        raise RuntimeError("GROQ_API_KEY not found in env or .env")
    return Groq(api_key=key)


def call_groq_with_timeout(client: Groq, messages: List[Dict[str, str]], timeout: int = 10):
    executor = ThreadPoolExecutor(max_workers=1)
    try:
        future = executor.submit(
            client.chat.completions.create,
            model=GROQ_MODEL,
            messages=messages,
            max_tokens=50,
            temperature=0,
        )
        try:
            return future.result(timeout=timeout)
        except FuturesTimeout:
            future.cancel()
            return None
    finally:
        executor.shutdown(wait=False, cancel_futures=True)


def call_groq(client: Groq, prompt: str, retries: int = 5, timeout_seconds: int = 10, complaint_id: Optional[str] = None) -> str:
    for i in range(retries):
        try:
            resp = call_groq_with_timeout(client, [{"role": "user", "content": prompt}], timeout=timeout_seconds)
            if resp is None:
                cid = complaint_id if complaint_id is not None else "unknown"
                log(f"Timeout on complaint {cid}, defaulting to FM")
                return ""
            return (resp.choices[0].message.content or "").strip()
        except Exception:
            if i == retries - 1:
                return ""
            time.sleep(1.2 * (i + 1))
    return ""


def parse_direct_response(text: str) -> str:
    t = norm(text).strip().upper()
    if t.startswith("FM"):
        return "FM"
    if t.startswith("PROJECT"):
        return "Project"
    if "FM" in t:
        return "FM"
    if "PROJECT" in t:
        return "Project"
    return "FM"


def parse_tier2(text: str):
    t = text.upper()
    if "PROJECT" in t:
        d = "Project"
    elif "FM" in t:
        d = "FM"
    else:
        d = "FM"
    m = re.search(r"CONFIDENCE\s*:\s*([0-9]*\.?[0-9]+)", t)
    if m:
        try:
            c = float(m.group(1))
        except Exception:
            c = 0.5
    else:
        nums = re.findall(r"\b([0-9]*\.?[0-9]+)\b", text)
        c = float(nums[0]) if nums else 0.5
    return d, max(0.0, min(1.0, c))


def parse_prob(text: str) -> float:
    s = norm(text)
    try:
        v = float(s)
        if 0 <= v <= 1:
            return v
    except Exception:
        pass
    m = re.search(r"([0-9]*\.?[0-9]+)", s)
    if m:
        try:
            v = float(m.group(1))
            if 0 <= v <= 1:
                return v
        except Exception:
            pass
    return 0.5


def sample_evenly(df: pd.DataFrame, max_n: int = 200) -> pd.DataFrame:
    if len(df) <= max_n:
        return df.sample(frac=1.0, random_state=42).reset_index(drop=True)
    groups = {k: g.sample(frac=1.0, random_state=42).copy() for k, g in df.groupby("ambiguous_category")}
    keys = sorted(groups.keys())
    base = max_n // len(keys)
    picks = []
    remain = max_n
    for k in keys:
        n = min(base, len(groups[k]))
        picks.append(groups[k].head(n))
        groups[k] = groups[k].iloc[n:]
        remain -= n
    while remain > 0:
        progressed = False
        for k in keys:
            if remain == 0:
                break
            if len(groups[k]) > 0:
                picks.append(groups[k].head(1))
                groups[k] = groups[k].iloc[1:]
                remain -= 1
                progressed = True
        if not progressed:
            break
    return pd.concat(picks, ignore_index=True).sample(frac=1.0, random_state=42).reset_index(drop=True)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    data_path = resolve_data_file()
    log(f"Using dataset: {data_path}")
    sheet, raw = pick_sheet(data_path)
    log(f"Using sheet: {sheet}")
    raw.columns = [str(c).strip() for c in raw.columns]

    col_text = "Complaint Title" if "Complaint Title" in raw.columns else raw.columns[0]
    col_label = "Issue Related To (FM/Project)" if "Issue Related To (FM/Project)" in raw.columns else raw.columns[0]
    col_flat = "Flat" if "Flat" in raw.columns else raw.columns[0]
    col_site = "Site Name" if "Site Name" in raw.columns else raw.columns[0]
    col_cat = "Sub Category" if "Sub Category" in raw.columns else ("Category" if "Category" in raw.columns else raw.columns[0])
    col_date = "Created Date" if "Created Date" in raw.columns else raw.columns[0]
    col_tid = "Ticket ID" if "Ticket ID" in raw.columns else None

    df = pd.DataFrame(
        {
            "ticket_id": raw[col_tid].astype(str) if col_tid else raw.index.astype(str),
            "text": raw[col_text].astype(str).fillna(""),
            "raw_label": raw[col_label].astype(str).fillna(""),
            "flat_id": raw[col_flat].astype(str).fillna(""),
            "community_id": raw[col_site].astype(str).fillna(""),
            "category": raw[col_cat].astype(str).fillna(""),
            "date": pd.to_datetime(raw[col_date], errors="coerce"),
        }
    )
    df = df[df["text"].str.strip() != ""].copy()
    df["label"] = df["raw_label"].map(map_label)
    df["ambiguous_category"] = df["category"].map(map_ambiguous_category)

    # remove pre-possession communities
    tmp = df.dropna(subset=["date"]).copy()
    tmp["month"] = tmp["date"].dt.to_period("M").astype(str)
    by_month = tmp.groupby(["community_id", "month"]).size().reset_index(name="cnt")
    avg_monthly = by_month.groupby("community_id")["cnt"].mean()
    mature = set(avg_monthly[avg_monthly >= 10].index.tolist())
    df_mature = df[df["community_id"].isin(mature)].copy()

    amb = df_mature[df_mature["ambiguous_category"].notna()].copy()
    eval_amb = sample_evenly(amb, max_n=200)
    eval_amb = eval_amb.reset_index(drop=True)
    eval_amb["eval_id"] = [f"amb_{i}" for i in range(len(eval_amb))]
    eval_path = OUT_DIR / "eval_ambiguous.csv"
    eval_amb.to_csv(eval_path, index=False)

    counts = eval_amb["ambiguous_category"].value_counts().sort_index()
    log("Ambiguous eval counts by category:")
    for k, v in counts.items():
        log(f"- {k}: {int(v)}")

    client = get_client()

    # Groq direct baseline v3
    log("\nRunning Groq direct baseline v3...")
    y_true = eval_amb["label"].tolist()
    y_pred = []
    raw_responses = []
    for idx, r in enumerate(eval_amb.itertuples(index=False), start=1):
        prompt = (
            "You must respond with exactly one word only. Classify this Indian residential maintenance complaint as FM or Project. "
            "FM means facility management responsibility: plumbing, electrical, housekeeping, security, pest control, routine maintenance. "
            "Project means developer warranty responsibility: structural defects, waterproofing failures, construction quality issues, post-possession defects. "
            f"Category: {r.category}. Complaint: {r.text}. Respond with one word only: FM or Project."
        )
        resp = call_groq(client, prompt, complaint_id=str(r.eval_id))
        raw_responses.append(resp)
        y_pred.append(parse_direct_response(resp))
        if idx % 20 == 0:
            log(f"Groq direct progress: {idx}/{len(eval_amb)}")

    direct_acc = float(accuracy_score(y_true, y_pred)) if y_true else 0.0
    direct_cwa = float(cost_weighted_accuracy(y_true, y_pred)) if y_true else 0.0
    direct_payload = {
        "overall_accuracy": direct_acc,
        "ambiguous_category_accuracy": direct_acc,
        "cost_weighted_accuracy": direct_cwa,
        "sample_size": len(eval_amb),
        "predictions": y_pred,
        "raw_responses": raw_responses,
        "eval_ids": eval_amb["eval_id"].tolist(),
    }
    (OUT_DIR / "results_groq_direct_v3.json").write_text(json.dumps(direct_payload, indent=2), encoding="utf-8")

    log("First 5 raw Groq responses:")
    for i, rr in enumerate(raw_responses[:5]):
        log(f"{i+1}. {rr}")
    log(f"Groq direct v3 metrics: accuracy={direct_acc:.4f}, cost_weighted_accuracy={direct_cwa:.4f}, sample_size={len(eval_amb)}")

    # history pool excludes eval complaints
    eval_ids = set(eval_amb["ticket_id"].astype(str).tolist())
    hist_pool = df_mature[~df_mature["ticket_id"].astype(str).isin(eval_ids)].copy()
    if hist_pool["date"].notna().any():
        hist_pool = hist_pool.sort_values("date")

    # Tier 2 on same set
    log("\nRunning ARIA Tier 2 on eval_ambiguous...")
    t2_rows = []
    for idx, r in enumerate(eval_amb.itertuples(index=False), start=1):
        h = hist_pool[hist_pool["flat_id"] == r.flat_id].tail(3)
        prior = "None" if len(h) == 0 else " | ".join([f"{x['category']}: {x['text']}" for _, x in h.iterrows()])
        prompt = (
            "You are an intelligent complaint routing system for a residential gated community in India. "
            "Determine whether this complaint is FM responsibility or Project responsibility. "
            f"Category: {r.category}. Complaint: {r.text}. Prior complaints from this flat: {prior}. "
            "Respond exactly in this format: Decision: FM or Project. Confidence: [number between 0 and 1]. Reasoning: one sentence."
        )
        resp = call_groq(client, prompt, complaint_id=str(r.eval_id))
        decision, confidence = parse_tier2(resp)
        t2_rows.append(
            {
                "eval_id": r.eval_id,
                "true_label": r.label,
                "category": r.category,
                "ambiguous_category": r.ambiguous_category,
                "decision": decision,
                "confidence": confidence,
                "escalated": confidence < 0.65,
                "prior": prior,
                "raw_response": resp,
            }
        )
        if idx % 20 == 0:
            log(f"Tier 2 progress: {idx}/{len(eval_amb)}")

    t2_df = pd.DataFrame(t2_rows)
    t2_acc = float(accuracy_score(t2_df["true_label"], t2_df["decision"])) if len(t2_df) else 0.0
    t2_cwa = float(cost_weighted_accuracy(t2_df["true_label"].tolist(), t2_df["decision"].tolist())) if len(t2_df) else 0.0
    t2_payload = {
        "overall_accuracy": t2_acc,
        "ambiguous_category_accuracy": t2_acc,
        "cost_weighted_accuracy": t2_cwa,
        "sample_size": len(t2_df),
        "escalated_count": int(t2_df["escalated"].sum()),
        "predictions": t2_rows,
    }
    (OUT_DIR / "results_aria_tier2_ambiguous.json").write_text(json.dumps(t2_payload, indent=2), encoding="utf-8")
    log(f"ARIA Tier 2 metrics: accuracy={t2_acc:.4f}, cost_weighted_accuracy={t2_cwa:.4f}, sample_size={len(t2_df)}")

    # Tier 3 on same set
    log("\nRunning ARIA Tier 3 on eval_ambiguous...")
    t3_rows = []
    for idx, r in enumerate(eval_amb.itertuples(index=False), start=1):
        h = hist_pool[hist_pool["flat_id"] == r.flat_id].tail(3)
        prior = "None" if len(h) == 0 else " | ".join([f"{x['category']}: {x['text']}" for _, x in h.iterrows()])
        p1 = (
            "You are analyzing a maintenance complaint to assess if it is a plumbing or pipe failure. "
            "This would be FM responsibility. Assess the probability between 0 and 1. "
            f"Category: {r.category}. Complaint: {r.text}. Prior history: {prior}. "
            "Respond with only a decimal number between 0 and 1. Nothing else."
        )
        p2 = (
            "You are analyzing a maintenance complaint to assess if it is a structural defect, waterproofing failure, or construction quality issue. "
            "This would be developer Project responsibility under warranty. Assess the probability between 0 and 1. "
            f"Category: {r.category}. Complaint: {r.text}. Prior history: {prior}. "
            "Respond with only a decimal number between 0 and 1. Nothing else."
        )
        p3 = (
            "You are analyzing a maintenance complaint to assess if it is an asset or equipment failure. "
            "This would be FM responsibility. Assess the probability between 0 and 1. "
            f"Category: {r.category}. Complaint: {r.text}. Prior history: {prior}. "
            "Respond with only a decimal number between 0 and 1. Nothing else."
        )
        r1 = call_groq(client, p1, complaint_id=f"{r.eval_id}_pipe")
        r2 = call_groq(client, p2, complaint_id=f"{r.eval_id}_struct")
        r3 = call_groq(client, p3, complaint_id=f"{r.eval_id}_asset")
        prob_pipe = parse_prob(r1)
        prob_struct = parse_prob(r2)
        prob_asset = parse_prob(r3)
        score_pipe = prob_pipe * 1.0
        score_struct = prob_struct * 10.0
        score_asset = prob_asset * 3.0
        decision = "Project" if score_struct >= max(score_pipe, score_asset) else "FM"
        t3_rows.append(
            {
                "eval_id": r.eval_id,
                "true_label": r.label,
                "category": r.category,
                "ambiguous_category": r.ambiguous_category,
                "pipe_probability": prob_pipe,
                "structural_probability": prob_struct,
                "asset_probability": prob_asset,
                "decision": decision,
                "raw_pipe": r1,
                "raw_structural": r2,
                "raw_asset": r3,
            }
        )
        if idx % 20 == 0:
            log(f"Tier 3 progress: {idx}/{len(eval_amb)}")
    t3_df = pd.DataFrame(t3_rows)
    t3_acc = float(accuracy_score(t3_df["true_label"], t3_df["decision"])) if len(t3_df) else 0.0
    t3_cwa = float(cost_weighted_accuracy(t3_df["true_label"].tolist(), t3_df["decision"].tolist())) if len(t3_df) else 0.0
    t3_payload = {
        "overall_accuracy": t3_acc,
        "ambiguous_category_accuracy": t3_acc,
        "cost_weighted_accuracy": t3_cwa,
        "sample_size": len(t3_df),
        "predictions": t3_rows,
    }
    (OUT_DIR / "results_aria_tier3_ambiguous.json").write_text(json.dumps(t3_payload, indent=2), encoding="utf-8")
    log(f"ARIA Tier 3 metrics: accuracy={t3_acc:.4f}, cost_weighted_accuracy={t3_cwa:.4f}, sample_size={len(t3_df)}")

    table = [
        ["Groq direct no context", f"{direct_acc:.4f}", f"{direct_cwa:.4f}", str(len(eval_amb))],
        ["ARIA Tier 2 with context", f"{t2_acc:.4f}", f"{t2_cwa:.4f}", str(len(t2_df))],
        ["ARIA Tier 3 cost-of-error", f"{t3_acc:.4f}", f"{t3_cwa:.4f}", str(len(t3_df))],
    ]
    lines = ["System | Accuracy on Ambiguous | Cost-Weighted Accuracy | Sample Size"]
    for r in table:
        lines.append(" | ".join(r))
    table_text = "\n".join(lines)
    (OUT_DIR / "ambiguous_comparison_final.txt").write_text(table_text + "\n", encoding="utf-8")

    log("\nFinal comparison table (ambiguous only):")
    log(table_text)
    log(f"\nSaved: {eval_path}")
    log(f"Saved: {OUT_DIR / 'results_groq_direct_v3.json'}")
    log(f"Saved: {OUT_DIR / 'results_aria_tier2_ambiguous.json'}")
    log(f"Saved: {OUT_DIR / 'results_aria_tier3_ambiguous.json'}")
    log(f"Saved: {OUT_DIR / 'ambiguous_comparison_final.txt'}")


if __name__ == "__main__":
    main()
