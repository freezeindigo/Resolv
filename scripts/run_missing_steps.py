import json
import os
import re
import time
from pathlib import Path
from typing import Dict, List

import pandas as pd
from groq import Groq
from sklearn.metrics import accuracy_score, confusion_matrix


OUT_DIR = Path.home() / "resolv-experiments" / "experiment_results"
MODEL = "llama-3.1-70b-versatile"
AMBIG_KEYS = ["leakage", "seepage", "civil", "carpentry", "common area", "mason"]


def log(msg: str):
    print(msg, flush=True)


def cost_weighted_accuracy(y_true, y_pred):
    total_cost = 0
    max_cost = 0
    for true, pred in zip(y_true, y_pred):
        if true == 'Project' and pred == 'FM':
            total_cost += 10
        elif true == 'FM' and pred == 'Project':
            total_cost += 1
        max_cost += 10
    return round(1 - (total_cost / max_cost), 4) if max_cost > 0 else 1.0


def norm(x) -> str:
    if pd.isna(x):
        return ""
    return str(x).strip()


def is_ambiguous(category: str) -> bool:
    c = norm(category).lower()
    return any(k in c for k in AMBIG_KEYS)


def parse_decision(text: str) -> str:
    t = norm(text).lower()
    if "project" in t:
        return "Project"
    if "fm" in t:
        return "FM"
    return "FM"


def parse_confidence(text: str) -> float:
    m = re.search(r"confidence\s*:\s*([0-9]*\.?[0-9]+)", norm(text).lower())
    if m:
        try:
            return max(0.0, min(1.0, float(m.group(1))))
        except Exception:
            return 0.5
    nums = re.findall(r"([0-9]*\.?[0-9]+)", norm(text))
    if nums:
        try:
            return max(0.0, min(1.0, float(nums[0])))
        except Exception:
            return 0.5
    return 0.5


def parse_prob(text: str) -> float:
    try:
        v = float(norm(text))
        return v if 0 <= v <= 1 else 0.5
    except Exception:
        m = re.search(r"([0-9]*\.?[0-9]+)", norm(text))
        if m:
            try:
                v = float(m.group(1))
                if 0 <= v <= 1:
                    return v
            except Exception:
                pass
    return 0.5


def get_col(df: pd.DataFrame, candidates: List[str], required=True) -> str:
    cols = list(df.columns)
    lc = {c.lower(): c for c in cols}
    for cand in candidates:
        for c in cols:
            if cand in c.lower():
                return c
    if required:
        raise ValueError(f"Could not find column for candidates={candidates} in {cols}")
    return ""


def compute_metrics(y_true: List[str], y_pred: List[str], cats: List[str]) -> Dict:
    overall = float(accuracy_score(y_true, y_pred)) if y_true else 0.0
    amb_idx = [i for i, c in enumerate(cats) if is_ambiguous(c)]
    if amb_idx:
        ay = [y_true[i] for i in amb_idx]
        ap = [y_pred[i] for i in amb_idx]
        amb = float(accuracy_score(ay, ap))
    else:
        amb = 0.0
    cwa = float(cost_weighted_accuracy(y_true, y_pred))
    cm = confusion_matrix(y_true, y_pred, labels=["FM", "Project"]).tolist() if y_true else [[0, 0], [0, 0]]
    return {
        "overall_accuracy": overall,
        "ambiguous_category_accuracy": amb,
        "cost_weighted_accuracy": cwa,
        "confusion_matrix_labels": ["FM", "Project"],
        "confusion_matrix": cm,
    }


def call_groq(client: Groq, prompt: str) -> str:
    for i in range(5):
        try:
            res = client.chat.completions.create(
                model=MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
            )
            out = res.choices[0].message.content or ""
            time.sleep(0.5)
            return out
        except Exception:
            if i == 4:
                time.sleep(0.5)
                return ""
            time.sleep(1.5 * (i + 1))
    time.sleep(0.5)
    return ""


def main():
    eval_path = OUT_DIR / "eval_500.csv"
    train_path = OUT_DIR / "train.csv"
    bert_text_path = OUT_DIR / "results_bert_text_only.json"
    bert_meta_path = OUT_DIR / "results_bert_text_meta.json"
    stats_path = OUT_DIR / "dataset_stats.json"
    step6_path = OUT_DIR / "results_groq_direct.json"
    step7_path = OUT_DIR / "results_aria_tier2.json"
    step8_path = OUT_DIR / "results_aria_tier3.json"
    step9_path = OUT_DIR / "results_aria_full.json"
    step10_path = OUT_DIR / "ablation_table.csv"
    step11_path = OUT_DIR / "paper_summary.txt"

    eval_df = pd.read_csv(eval_path)
    train_df = pd.read_csv(train_path)
    with open(bert_text_path, "r", encoding="utf-8") as f:
        bert_text = json.load(f)
    with open(bert_meta_path, "r", encoding="utf-8") as f:
        bert_meta = json.load(f)
    with open(stats_path, "r", encoding="utf-8") as f:
        ds_stats = json.load(f)

    text_col = get_col(eval_df, ["text", "complaint", "description", "issue"])
    label_col = get_col(eval_df, ["label", "ownership", "issue related"])
    cat_col = get_col(eval_df, ["category", "sub category", "trade", "type"])
    flat_col = get_col(eval_df, ["flat", "unit", "apartment", "door"])
    train_text_col = get_col(train_df, ["text", "complaint", "description", "issue"])
    train_cat_col = get_col(train_df, ["category", "sub category", "trade", "type"])
    train_flat_col = get_col(train_df, ["flat", "unit", "apartment", "door"])

    if "date" in train_df.columns:
        train_df["_sort_date"] = pd.to_datetime(train_df["date"], errors="coerce")
        train_df = train_df.sort_values("_sort_date")

    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError("GROQ_API_KEY not set in environment")
    client = Groq(api_key=api_key)

    # STEP 6
    if step6_path.exists():
        with open(step6_path, "r", encoding="utf-8") as f:
            step6 = json.load(f)
        log("STEP 6 already exists. Loaded existing results_groq_direct.json")
    else:
        log("STEP 6: Running Groq direct routing on 100 ambiguous complaints...")
        amb_df = eval_df[eval_df[cat_col].map(is_ambiguous)].copy().reset_index(drop=True)
        conf = bert_meta.get("prediction_confidence_scores", [])
        if isinstance(conf, list) and len(conf) == len(eval_df):
            eval_tmp = eval_df.copy().reset_index(drop=True)
            eval_tmp["_conf"] = conf
            target = eval_tmp[eval_tmp[cat_col].map(is_ambiguous)].sort_values("_conf").head(100).copy()
        else:
            target = amb_df.head(100).copy()
        preds = []
        for _, r in target.iterrows():
            prompt = (
                "You are a facility management routing expert for Indian residential gated communities. "
                "Classify this maintenance complaint as FM (facility management responsibility: routine maintenance, plumbing, electrical, housekeeping, security, pest control) "
                "or Project (developer warranty responsibility: structural defects, waterproofing failures, construction quality issues, post-possession defects). "
                f"Category: {r[cat_col]}. Complaint: {r[text_col]}. Respond with only FM or Project."
            )
            resp = call_groq(client, prompt)
            preds.append(parse_decision(resp))
        y_true = target[label_col].astype(str).tolist()
        y_pred = preds
        step6 = compute_metrics(y_true, y_pred, target[cat_col].astype(str).tolist())
        step6["sample_size"] = int(len(target))
        step6["predictions"] = y_pred
        step6["indices"] = target.index.tolist()
        with open(step6_path, "w", encoding="utf-8") as f:
            json.dump(step6, f, indent=2)
        log(f"STEP 6 complete. Overall={step6['overall_accuracy']:.4f}, CWA={step6['cost_weighted_accuracy']:.4f}")

    # STEP 7
    if step7_path.exists():
        with open(step7_path, "r", encoding="utf-8") as f:
            step7 = json.load(f)
        log("STEP 7 already exists. Loaded existing results_aria_tier2.json")
    else:
        log("STEP 7: Running ARIA Tier 2 on full eval_500...")
        predictions = []
        for idx, r in eval_df.reset_index(drop=True).iterrows():
            hist = train_df[train_df[train_flat_col].astype(str) == str(r[flat_col])].tail(3)
            if len(hist) == 0:
                prior = "None"
            else:
                prior = " | ".join([f"{h[train_cat_col]}: {h[train_text_col]}" for _, h in hist.iterrows()])
            prompt = (
                "You are an intelligent complaint routing system for a residential gated community in India. "
                "Determine if this complaint is FM (facility management) or Project (developer warranty) responsibility. "
                f"Category: {r[cat_col]}. Complaint: {r[text_col]}. Prior complaints from this flat: {prior}. "
                "Respond in this exact format: Decision: FM or Project. Confidence: 0.XX. Reasoning: one sentence."
            )
            resp = call_groq(client, prompt)
            decision = parse_decision(resp)
            confidence = parse_confidence(resp)
            escalated = bool(confidence < 0.65)
            predictions.append(
                {
                    "index": int(idx),
                    "category": str(r[cat_col]),
                    "text": str(r[text_col]),
                    "true_label": str(r[label_col]),
                    "decision": decision,
                    "confidence": float(confidence),
                    "escalated": escalated,
                    "prior": prior,
                }
            )
        pred_df = pd.DataFrame(predictions)
        step7 = compute_metrics(
            pred_df["true_label"].tolist(),
            pred_df["decision"].tolist(),
            pred_df["category"].tolist(),
        )
        step7["total_eval_count"] = int(len(pred_df))
        step7["escalated_count"] = int(pred_df["escalated"].sum())
        step7["predictions"] = predictions
        with open(step7_path, "w", encoding="utf-8") as f:
            json.dump(step7, f, indent=2)
        log(
            f"STEP 7 complete. Overall={step7['overall_accuracy']:.4f}, "
            f"Amb={step7['ambiguous_category_accuracy']:.4f}, CWA={step7['cost_weighted_accuracy']:.4f}, "
            f"Escalated={step7['escalated_count']}"
        )

    # STEP 8
    if step8_path.exists():
        with open(step8_path, "r", encoding="utf-8") as f:
            step8 = json.load(f)
        log("STEP 8 already exists. Loaded existing results_aria_tier3.json")
    else:
        log("STEP 8: Running ARIA Tier 3 on ambiguous complaints...")
        pred_map = {int(p["index"]): p for p in step7["predictions"]}
        amb_eval = eval_df.reset_index(drop=True)
        amb_eval = amb_eval[amb_eval[cat_col].map(is_ambiguous)].copy()
        predictions = []
        changed = 0
        changed_match = 0
        for idx, r in amb_eval.iterrows():
            prior = pred_map.get(int(idx), {}).get("prior", "None")
            pa = call_groq(
                client,
                f"Probability 0 to 1 that this is a plumbing or pipe failure (FM responsibility). Category: {r[cat_col]}. Complaint: {r[text_col]}. History: {prior}. Output only a decimal number.",
            )
            pb = call_groq(
                client,
                f"Probability 0 to 1 that this is a structural defect or waterproofing failure (Project developer responsibility). Category: {r[cat_col]}. Complaint: {r[text_col]}. History: {prior}. Output only a decimal number.",
            )
            pc = call_groq(
                client,
                f"Probability 0 to 1 that this is an asset or equipment failure (FM responsibility). Category: {r[cat_col]}. Complaint: {r[text_col]}. History: {prior}. Output only a decimal number.",
            )
            a = parse_prob(pa)
            b = parse_prob(pb)
            c = parse_prob(pc)

            score_pipe = a * 1.0
            score_structural = b * 10.0
            score_asset = c * 3.0
            weighted = "Project" if score_structural >= max(score_pipe, score_asset) else "FM"

            raw_max = max(a, b, c)
            pure = "Project" if raw_max == b else "FM"
            if weighted != pure:
                changed += 1
                if weighted == str(r[label_col]):
                    changed_match += 1
            predictions.append(
                {
                    "index": int(idx),
                    "category": str(r[cat_col]),
                    "text": str(r[text_col]),
                    "true_label": str(r[label_col]),
                    "pipe_probability": a,
                    "structural_probability": b,
                    "asset_probability": c,
                    "weighted_decision": weighted,
                    "pure_probability_decision": pure,
                    "prior": prior,
                }
            )
        p8 = pd.DataFrame(predictions)
        step8 = compute_metrics(
            p8["true_label"].tolist(),
            p8["weighted_decision"].tolist(),
            p8["category"].tolist(),
        )
        step8["sample_size"] = int(len(p8))
        step8["cost_arbiter_changed_decisions"] = int(changed)
        step8["changed_decisions_matching_ground_truth"] = int(changed_match)
        step8["predictions"] = predictions
        with open(step8_path, "w", encoding="utf-8") as f:
            json.dump(step8, f, indent=2)
        log(
            f"STEP 8 complete. Overall={step8['overall_accuracy']:.4f}, "
            f"Amb={step8['ambiguous_category_accuracy']:.4f}, CWA={step8['cost_weighted_accuracy']:.4f}, "
            f"Changed={step8['cost_arbiter_changed_decisions']}/{step8['sample_size']}"
        )

    # STEP 9
    if step9_path.exists():
        with open(step9_path, "r", encoding="utf-8") as f:
            step9 = json.load(f)
        log("STEP 9 already exists. Loaded existing results_aria_full.json")
    else:
        log("STEP 9: Building ARIA full pipeline result...")
        t2 = {int(p["index"]): p for p in step7["predictions"]}
        t3 = {int(p["index"]): p for p in step8["predictions"]}
        final = []
        for i, r in eval_df.reset_index(drop=True).iterrows():
            t2p = t2.get(int(i))
            if t2p and t2p.get("escalated") and int(i) in t3:
                dec = t3[int(i)]["weighted_decision"]
                source = "tier3"
            else:
                dec = t2p["decision"] if t2p else "FM"
                source = "tier2"
            final.append(
                {
                    "index": int(i),
                    "category": str(r[cat_col]),
                    "true_label": str(r[label_col]),
                    "decision": dec,
                    "source": source,
                }
            )
        ff = pd.DataFrame(final)
        step9 = compute_metrics(ff["true_label"].tolist(), ff["decision"].tolist(), ff["category"].tolist())
        step9["predictions"] = final
        with open(step9_path, "w", encoding="utf-8") as f:
            json.dump(step9, f, indent=2)
        log(
            f"STEP 9 complete. Overall={step9['overall_accuracy']:.4f}, "
            f"Amb={step9['ambiguous_category_accuracy']:.4f}, CWA={step9['cost_weighted_accuracy']:.4f}"
        )

    # STEP 10
    log("STEP 10: Compiling ablation table...")
    rows = [
        {
            "System": "BERT text-only",
            "Overall Accuracy": bert_text.get("overall_accuracy", 0.0),
            "Ambiguous Category Accuracy": bert_text.get("ambiguous_category_accuracy", 0.0),
            "Cost-Weighted Accuracy": bert_text.get("cost_weighted_accuracy", 0.0),
            "Notes": "No context no metadata",
        },
        {
            "System": "BERT text plus metadata",
            "Overall Accuracy": bert_meta.get("overall_accuracy", 0.0),
            "Ambiguous Category Accuracy": bert_meta.get("ambiguous_category_accuracy", 0.0),
            "Cost-Weighted Accuracy": bert_meta.get("cost_weighted_accuracy", 0.0),
            "Notes": "Category tag added",
        },
        {
            "System": "Groq direct routing",
            "Overall Accuracy": step6.get("overall_accuracy", 0.0),
            "Ambiguous Category Accuracy": step6.get("ambiguous_category_accuracy", 0.0),
            "Cost-Weighted Accuracy": step6.get("cost_weighted_accuracy", 0.0),
            "Notes": "Strong LLM no building context",
        },
        {
            "System": "ARIA Tier 2 context-aware",
            "Overall Accuracy": step7.get("overall_accuracy", 0.0),
            "Ambiguous Category Accuracy": step7.get("ambiguous_category_accuracy", 0.0),
            "Cost-Weighted Accuracy": step7.get("cost_weighted_accuracy", 0.0),
            "Notes": "Flat history context added",
        },
        {
            "System": "ARIA full pipeline",
            "Overall Accuracy": step9.get("overall_accuracy", 0.0),
            "Ambiguous Category Accuracy": step9.get("ambiguous_category_accuracy", 0.0),
            "Cost-Weighted Accuracy": step9.get("cost_weighted_accuracy", 0.0),
            "Notes": "Three tier with cost of error arbiter",
        },
    ]
    ab = pd.DataFrame(rows)
    ab.to_csv(step10_path, index=False)
    log(ab.to_string(index=False))
    log("STEP 10 complete.")

    # STEP 11
    log("STEP 11: Writing paper summary...")
    changed = step8.get("cost_arbiter_changed_decisions", 0)
    total = step8.get("sample_size", 0)
    matched = step8.get("changed_decisions_matching_ground_truth", 0)
    pct = (matched / changed * 100) if changed else 0.0
    summary = []
    summary.append("DATASET STATISTICS")
    summary.append(json.dumps(ds_stats, indent=2))
    summary.append("")
    summary.append("RESULTS TABLE")
    summary.append(ab.to_string(index=False))
    summary.append("")
    summary.append("KEY FINDINGS")
    summary.append(
        f"Cost-of-error arbiter changed routing in {changed} of {total} ambiguous complaints versus pure probability."
    )
    summary.append(
        f"Of those {changed} changed decisions, {matched} matched ground truth ({pct:.2f} percent)."
    )
    summary.append(
        f"ARIA full pipeline: {step9.get('overall_accuracy', 0.0)*100:.2f} percent overall accuracy, "
        f"{step9.get('ambiguous_category_accuracy', 0.0)*100:.2f} percent ambiguous accuracy, "
        f"{step9.get('cost_weighted_accuracy', 0.0)*100:.2f} percent cost-weighted accuracy."
    )
    summary.append(
        f"BERT text-only baseline: {bert_text.get('overall_accuracy', 0.0)*100:.2f} percent overall, "
        f"{bert_text.get('ambiguous_category_accuracy', 0.0)*100:.2f} percent ambiguous, "
        f"{bert_text.get('cost_weighted_accuracy', 0.0)*100:.2f} percent cost-weighted."
    )
    summary.append(
        f"Groq direct routing no context: {step6.get('overall_accuracy', 0.0)*100:.2f} percent overall, "
        f"{step6.get('ambiguous_category_accuracy', 0.0)*100:.2f} percent ambiguous, "
        f"{step6.get('cost_weighted_accuracy', 0.0)*100:.2f} percent cost-weighted."
    )
    summary_text = "\n".join(summary)
    with open(step11_path, "w", encoding="utf-8") as f:
        f.write(summary_text)
    log(summary_text)
    log("STEP 11 complete.")
    log("All missing steps complete.")


if __name__ == "__main__":
    main()
