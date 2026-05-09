import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from groq import Groq
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)


MODEL_NAME = "bert-base-uncased"
GROQ_MODEL = "llama-3.1-70b-versatile"
AMBIGUOUS_CATEGORIES = {"leakage", "seepage", "civil work", "carpentry", "common area", "mason"}
OUTPUT_DIR = Path.home() / "resolv-experiments" / "experiment_results"
DATA_FILE = Path.home() / "Downloads" / "GPL complaint data.xlsx"


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


def log(msg: str) -> None:
    print(msg, flush=True)


def norm_text(x) -> str:
    if pd.isna(x):
        return ""
    return str(x).strip()


def clean_category(x) -> str:
    return re.sub(r"\s+", " ", norm_text(x).lower())


def is_ambiguous(category: str) -> bool:
    c = clean_category(category)
    return any(a in c for a in AMBIGUOUS_CATEGORIES)


def safe_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, default=str)


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


def choose_column(columns: List[str], keywords: List[str], preferred_values: Optional[pd.Series] = None) -> str:
    scores = []
    for col in columns:
        lc = col.lower()
        score = 0
        for i, k in enumerate(keywords):
            if k in lc:
                score += (len(keywords) - i) * 10
        if preferred_values is not None:
            uniq = preferred_values[col].dropna().astype(str).str.lower().value_counts().head(20).index.tolist()
            uniq_joined = " ".join(uniq)
            for k in ["fm", "facility", "project", "developer", "owner", "warranty", "dlp"]:
                if k in uniq_joined:
                    score += 7
        scores.append((score, col))
    scores.sort(reverse=True)
    return scores[0][1]


def map_labels(raw_label: str) -> str:
    t = norm_text(raw_label).lower()
    if t == "":
        return "FM"
    project_terms = [
        "project",
        "developer",
        "construction",
        "structural",
        "warranty",
        "waterproof",
        "dlp",
        "builder",
    ]
    fm_terms = [
        "fm",
        "facility",
        "maintenance",
        "housekeeping",
        "security",
        "electrical",
        "plumbing",
        "operations",
    ]
    if any(k in t for k in project_terms):
        return "Project"
    if any(k in t for k in fm_terms):
        return "FM"
    if t in {"owner", "owners"}:
        return "Project"
    return "FM"


class ComplaintDataset(Dataset):
    def __init__(self, texts: List[str], labels: List[int], tokenizer, max_len=256):
        self.enc = tokenizer(texts, truncation=True, padding=True, max_length=max_len)
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {k: torch.tensor(v[idx]) for k, v in self.enc.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item


def compute_metrics_from_preds(y_true: List[str], y_pred: List[str], categories: List[str]) -> Dict:
    overall_acc = accuracy_score(y_true, y_pred) if y_true else 0.0
    cwa = cost_weighted_accuracy(y_true, y_pred) if y_true else 0.0
    amb_idx = [i for i, c in enumerate(categories) if is_ambiguous(c)]
    if amb_idx:
        amb_true = [y_true[i] for i in amb_idx]
        amb_pred = [y_pred[i] for i in amb_idx]
        amb_acc = accuracy_score(amb_true, amb_pred)
    else:
        amb_acc = 0.0
    cm = confusion_matrix(y_true, y_pred, labels=["FM", "Project"]).tolist() if y_true else [[0, 0], [0, 0]]
    return {
        "overall_accuracy": float(overall_acc),
        "ambiguous_category_accuracy": float(amb_acc),
        "cost_weighted_accuracy": float(cwa),
        "confusion_matrix_labels": ["FM", "Project"],
        "confusion_matrix": cm,
    }


def run_bert(train_df: pd.DataFrame, eval_df: pd.DataFrame, use_meta: bool, out_json: Path) -> Dict:
    if len(train_df) > 4000:
        train_df = (
            train_df.groupby(["label", "category_clean"], group_keys=False)
            .apply(lambda x: x.sample(min(len(x), max(1, int(4000 * len(x) / len(train_df)))), random_state=42))
            .reset_index(drop=True)
        )

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)

    if use_meta:
        train_texts = [f"Category: {c}. Complaint: {t}" for c, t in zip(train_df["category"], train_df["text"])]
        eval_texts = [f"Category: {c}. Complaint: {t}" for c, t in zip(eval_df["category"], eval_df["text"])]
    else:
        train_texts = train_df["text"].astype(str).tolist()
        eval_texts = eval_df["text"].astype(str).tolist()

    label_map = {"FM": 0, "Project": 1}
    train_labels = [label_map[x] for x in train_df["label"]]
    eval_labels = [label_map[x] for x in eval_df["label"]]

    train_ds = ComplaintDataset(train_texts, train_labels, tokenizer)
    eval_ds = ComplaintDataset(eval_texts, eval_labels, tokenizer)

    device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    args = TrainingArguments(
        output_dir=str(OUTPUT_DIR / ("bert_meta_ckpt" if use_meta else "bert_text_ckpt")),
        num_train_epochs=3,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        learning_rate=2e-5,
        logging_steps=50,
        disable_tqdm=True,
        save_strategy="no",
        eval_strategy="no",
        report_to=[],
    )
    trainer = Trainer(model=model, args=args, train_dataset=train_ds, tokenizer=tokenizer)
    trainer.train()
    pred_output = trainer.predict(eval_ds)
    logits = pred_output.predictions
    probs = torch.softmax(torch.tensor(logits), dim=1).numpy()
    pred_ids = np.argmax(probs, axis=1)
    inv = {0: "FM", 1: "Project"}
    y_pred = [inv[i] for i in pred_ids]
    y_true = eval_df["label"].tolist()
    confidence = probs.max(axis=1).tolist()

    metrics = compute_metrics_from_preds(y_true, y_pred, eval_df["category"].tolist())
    metrics["prediction_confidence_scores"] = confidence
    metrics["predictions"] = y_pred
    safe_json(out_json, metrics)
    return metrics


def groq_client() -> Groq:
    key = os.getenv("GROQ_API_KEY")
    if not key:
        raise RuntimeError("GROQ_API_KEY is not set in environment.")
    return Groq(api_key=key)


def call_groq(client: Groq, prompt: str, max_retries: int = 5) -> str:
    for i in range(max_retries):
        try:
            resp = client.chat.completions.create(
                model=GROQ_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
            )
            return resp.choices[0].message.content or ""
        except Exception:
            if i == max_retries - 1:
                return ""
            time.sleep(1.5 * (i + 1))
    return ""


def parse_decision(text: str) -> str:
    t = norm_text(text).lower()
    if "project" in t:
        return "Project"
    if "fm" in t:
        return "FM"
    return "FM"


def parse_confidence(text: str) -> float:
    m = re.search(r"confidence\s*:\s*([0-9]*\.?[0-9]+)", text.lower())
    if m:
        try:
            return float(m.group(1))
        except Exception:
            return 0.5
    nums = re.findall(r"\b([0-9]*\.?[0-9]+)\b", text)
    if nums:
        try:
            return float(nums[0])
        except Exception:
            return 0.5
    return 0.5


def parse_prob(text: str) -> float:
    try:
        v = float(norm_text(text))
        if 0 <= v <= 1:
            return v
        return 0.5
    except Exception:
        m = re.search(r"([0-9]*\.?[0-9]+)", norm_text(text))
        if m:
            try:
                v = float(m.group(1))
                if 0 <= v <= 1:
                    return v
            except Exception:
                pass
    return 0.5


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    log("Running experiment pipeline...")

    # STEP 0
    log("\nSTEP 0: LOAD AND INSPECT DATA")
    data_file = resolve_data_file()
    log(f"Using data file: {data_file}")
    sheet_name, raw_df = pick_sheet(data_file)
    raw_df.columns = [str(c).strip() for c in raw_df.columns]
    log(f"Selected sheet: {sheet_name}")
    log(f"Columns: {list(raw_df.columns)}")
    log("First 5 rows:")
    log(raw_df.head(5).to_string(index=False))

    cols = raw_df.columns.tolist()
    complaint_col = choose_column(cols, ["description", "complaint", "issue", "text", "remarks"])
    label_col = choose_column(cols, ["ownership", "owner", "fm", "project", "category", "type"], raw_df)
    flat_col = choose_column(cols, ["flat", "unit", "apartment", "door", "house"])
    comm_col = choose_column(cols, ["community", "site", "society", "project", "tower", "block"])
    cat_col = choose_column(cols, ["category", "trade", "type", "issue"])
    date_col = choose_column(cols, ["date", "logged", "created", "opened", "timestamp"])

    mapped = {
        "complaint_text_column": complaint_col,
        "ownership_label_column": label_col,
        "flat_unit_column": flat_col,
        "community_site_column": comm_col,
        "complaint_category_column": cat_col,
        "date_column": date_col,
    }
    log("Column mapping:")
    log(json.dumps(mapped, indent=2))

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
    df["date"] = pd.to_datetime(df["date_raw"], errors="coerce")
    df["label"] = df["raw_label"].map(map_labels)

    uniq_raw = sorted(set(df["raw_label"].dropna().astype(str).str.strip().tolist()))
    label_mapping_preview = {u: map_labels(u) for u in uniq_raw[:100]}
    log("Label mapping preview:")
    log(json.dumps(label_mapping_preview, indent=2))

    df = df[df["text"].str.strip() != ""].copy()
    df["category_clean"] = df["category"].map(clean_category)
    df["is_ambiguous"] = df["category"].map(is_ambiguous)
    log("STEP 0 complete.")

    # STEP 1
    log("\nSTEP 1: DATASET STATISTICS")
    total = len(df)
    total_communities = df["community_id"].nunique()
    date_min = df["date"].min()
    date_max = df["date"].max()
    split = (df["label"].value_counts(normalize=True) * 100).to_dict()

    amb_stats = {}
    for ac in AMBIGUOUS_CATEGORIES:
        sub = df[df["category_clean"].str.contains(ac, na=False)]
        vc = (sub["label"].value_counts(normalize=True) * 100).to_dict() if len(sub) else {}
        amb_stats[ac] = {"count": int(len(sub)), "FM_percent": float(vc.get("FM", 0.0)), "Project_percent": float(vc.get("Project", 0.0))}

    combo = df.groupby(["flat_id", "category_clean"]).size()
    repeat_rate = float((combo[combo > 1].sum() / len(df) * 100) if len(df) else 0.0)
    combo_5_plus = int((combo >= 5).sum())
    med_age = None
    if df["date"].notna().any():
        med_age = float((pd.Timestamp.now() - df["date"]).dt.days.median())

    stats = {
        "total_complaints": int(total),
        "total_communities": int(total_communities),
        "date_range": {"earliest": str(date_min.date()) if pd.notna(date_min) else None, "latest": str(date_max.date()) if pd.notna(date_max) else None},
        "overall_split_percent": {"FM": float(split.get("FM", 0.0)), "Project": float(split.get("Project", 0.0))},
        "ambiguous_category_splits": amb_stats,
        "repeat_complaint_rate_percent": repeat_rate,
        "flat_category_combinations_5_or_more": combo_5_plus,
        "median_complaint_age_days": med_age,
    }
    safe_json(OUTPUT_DIR / "dataset_stats.json", stats)
    log(json.dumps(stats, indent=2))
    log("STEP 1 complete.")

    # STEP 2
    log("\nSTEP 2: CLEAN DATA AND CREATE SPLITS")
    removed = []
    if df["date"].notna().any():
        tmp = df.dropna(subset=["date"]).copy()
        tmp["month"] = tmp["date"].dt.to_period("M").astype(str)
        by_month = tmp.groupby(["community_id", "month"]).size().reset_index(name="cnt")
        avg_monthly = by_month.groupby("community_id")["cnt"].mean()
        mature_comms = avg_monthly[avg_monthly >= 10].index
        removed_comms = avg_monthly[avg_monthly < 10]
        removed = [{"community_id": k, "avg_complaints_per_month": float(v), "reason": "avg complaints/month < 10"} for k, v in removed_comms.items()]
        df_mature = df[df["community_id"].isin(mature_comms)].copy()
    else:
        df_mature = df.copy()
    if len(df_mature) < 600:
        df_mature = df.copy()
        removed = []
        log("Fallback: mature-community filter would over-shrink data; using all communities.")
    log(f"Removed communities: {json.dumps(removed, indent=2)}")

    df_mature["strata"] = df_mature["label"] + "||" + df_mature["category_clean"]
    test_size = min(500, len(df_mature))
    if test_size == len(df_mature):
        eval_df = df_mature.sample(n=test_size, random_state=42).copy()
        train_df = df_mature.drop(eval_df.index).copy()
    else:
        try:
            train_df, eval_df = train_test_split(df_mature, test_size=test_size, random_state=42, stratify=df_mature["strata"])
        except Exception:
            train_df, eval_df = train_test_split(df_mature, test_size=test_size, random_state=42, stratify=df_mature["label"])

    train_df.to_csv(OUTPUT_DIR / "train.csv", index=False)
    eval_df.to_csv(OUTPUT_DIR / "eval_500.csv", index=False)
    tr_split = (train_df["label"].value_counts(normalize=True) * 100).to_dict() if len(train_df) else {}
    ev_split = (eval_df["label"].value_counts(normalize=True) * 100).to_dict() if len(eval_df) else {}
    log(f"train size={len(train_df)}, eval size={len(eval_df)}")
    log(f"train FM/Project ratio: FM={tr_split.get('FM', 0):.2f}% Project={tr_split.get('Project', 0):.2f}%")
    log(f"eval FM/Project ratio: FM={ev_split.get('FM', 0):.2f}% Project={ev_split.get('Project', 0):.2f}%")
    log("STEP 2 complete.")

    # STEP 3
    log("\nSTEP 3: COST-WEIGHTED ACCURACY FUNCTION TEST")
    cwa_test = cost_weighted_accuracy(["Project", "FM", "FM", "Project"], ["FM", "FM", "Project", "Project"])
    log(f"Test result: {cwa_test:.4f}")
    log("STEP 3 complete.")

    # STEP 4
    log("\nSTEP 4: BERT TEXT-ONLY BASELINE")
    bert_text_metrics = run_bert(train_df, eval_df, use_meta=False, out_json=OUTPUT_DIR / "results_bert_text_only.json")
    log(json.dumps({k: v for k, v in bert_text_metrics.items() if k != "prediction_confidence_scores" and k != "predictions"}, indent=2))
    log("STEP 4 complete.")

    # STEP 5
    log("\nSTEP 5: BERT TEXT PLUS METADATA BASELINE")
    bert_meta_metrics = run_bert(train_df, eval_df, use_meta=True, out_json=OUTPUT_DIR / "results_bert_text_meta.json")
    log(json.dumps({k: v for k, v in bert_meta_metrics.items() if k != "prediction_confidence_scores" and k != "predictions"}, indent=2))
    log("STEP 5 complete.")

    client = groq_client()

    # STEP 6
    log("\nSTEP 6: GROQ DIRECT ROUTING BASELINE")
    conf = np.array(bert_meta_metrics["prediction_confidence_scores"])
    eval_df2 = eval_df.copy().reset_index(drop=True)
    eval_df2["confidence"] = conf[: len(eval_df2)]
    cand = eval_df2[eval_df2["category"].map(is_ambiguous)].sort_values("confidence", ascending=True).head(100).copy()
    y_true = cand["label"].tolist()
    y_pred = []
    for _, r in cand.iterrows():
        prompt = (
            "You are a facility management routing expert for Indian residential gated communities. "
            "A maintenance complaint has been logged. Determine whether it is the responsibility of FM "
            "(facility management company: routine maintenance, plumbing, electrical, housekeeping, security, pest control) "
            "or Project (developer or warranty team: structural defects, construction quality issues, post-possession defects, "
            f"waterproofing failures, DLP items). Category: {r['category']}. Complaint: {r['text']}. "
            "Respond in exactly this format: Decision: FM or Project. Reasoning: one sentence explaining why."
        )
        resp = call_groq(client, prompt)
        y_pred.append(parse_decision(resp))
    metrics6 = compute_metrics_from_preds(y_true, y_pred, cand["category"].tolist())
    metrics6["sample_size"] = len(cand)
    metrics6["predictions"] = y_pred
    safe_json(OUTPUT_DIR / "results_groq_direct.json", metrics6)
    log(json.dumps({k: v for k, v in metrics6.items() if k != "predictions"}, indent=2))
    log("STEP 6 complete.")

    # STEP 7
    log("\nSTEP 7: ARIA TIER 2 SIMULATION")
    train_hist = train_df.copy()
    if train_hist["date"].notna().any():
        train_hist = train_hist.sort_values("date")
    preds7 = []
    for _, r in eval_df2.iterrows():
        hist = train_hist[train_hist["flat_id"] == r["flat_id"]].tail(3)
        if len(hist) == 0:
            prior = "None"
        else:
            prior = " | ".join([f"{x['category']}: {x['text']}" for _, x in hist.iterrows()])
        prompt = (
            "You are an intelligent complaint routing system for a residential gated community in India. "
            "Your job is to determine whether this complaint is the responsibility of the FM company (routine maintenance) "
            "or the Project team (developer warranty and structural issues). Use the complaint details and prior history from this flat "
            f"to make your decision. Category: {r['category']}. Complaint: {r['text']}. Prior complaints from this flat: {prior}. "
            "Think step by step about what could cause this issue and who is responsible. Respond in exactly this format: "
            "Decision: FM or Project. Confidence: [number between 0 and 1]. Reasoning: one sentence."
        )
        resp = call_groq(client, prompt)
        decision = parse_decision(resp)
        confidence = parse_confidence(resp)
        escalated = confidence < 0.65
        preds7.append(
            {
                "index": int(r.name),
                "category": r["category"],
                "text": r["text"],
                "true_label": r["label"],
                "decision": decision,
                "confidence": float(confidence),
                "escalated": bool(escalated),
                "prior": prior,
            }
        )
    df7 = pd.DataFrame(preds7)
    non_esc = df7[~df7["escalated"]].copy()
    m7 = compute_metrics_from_preds(non_esc["true_label"].tolist(), non_esc["decision"].tolist(), non_esc["category"].tolist())
    m7["evaluated_non_escalated_count"] = int(len(non_esc))
    m7["escalated_count"] = int(df7["escalated"].sum())
    m7["total_eval_count"] = int(len(df7))
    m7["predictions"] = preds7
    safe_json(OUTPUT_DIR / "results_aria_tier2.json", m7)
    log(json.dumps({k: v for k, v in m7.items() if k != "predictions"}, indent=2))
    log("STEP 7 complete.")

    # STEP 8
    log("\nSTEP 8: ARIA TIER 3 COST OF ERROR SIMULATION")
    esc_idx = set(df7[df7["escalated"]]["index"].tolist())
    amb_idx = set(eval_df2[eval_df2["category"].map(is_ambiguous)].index.tolist())
    idx8 = sorted(esc_idx.union(amb_idx))
    target8 = eval_df2.loc[idx8].copy()
    prior_map = {p["index"]: p["prior"] for p in preds7}

    predictions8 = []
    changed = 0
    changed_correct = 0
    for _, r in target8.iterrows():
        prior = prior_map.get(int(r.name), "None")
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
            futures = [ex.submit(call_groq, client, p) for p in [p1, p2, p3]]
            resp_pipe, resp_struct, resp_asset = [f.result() for f in futures]
        prob_pipe = parse_prob(resp_pipe)
        prob_struct = parse_prob(resp_struct)
        prob_asset = parse_prob(resp_asset)

        score_pipe = prob_pipe * 1.0
        score_struct = prob_struct * 10.0
        score_asset = prob_asset * 3.0
        weighted_dec = "Project" if score_struct >= max(score_pipe, score_asset) else "FM"

        raw_max = max(prob_pipe, prob_struct, prob_asset)
        if raw_max == prob_struct:
            pure_dec = "Project"
        else:
            pure_dec = "FM"

        if weighted_dec != pure_dec:
            changed += 1
            if weighted_dec == r["label"]:
                changed_correct += 1

        predictions8.append(
            {
                "index": int(r.name),
                "category": r["category"],
                "text": r["text"],
                "true_label": r["label"],
                "pipe_probability": prob_pipe,
                "structural_probability": prob_struct,
                "asset_probability": prob_asset,
                "weighted_decision": weighted_dec,
                "pure_probability_decision": pure_dec,
            }
        )

    df8 = pd.DataFrame(predictions8)
    m8 = compute_metrics_from_preds(df8["true_label"].tolist(), df8["weighted_decision"].tolist(), df8["category"].tolist())
    m8["sample_size"] = int(len(df8))
    m8["cost_arbiter_changed_decisions"] = int(changed)
    m8["changed_decisions_matching_ground_truth"] = int(changed_correct)
    m8["predictions"] = predictions8
    safe_json(OUTPUT_DIR / "results_aria_tier3.json", m8)
    log(json.dumps({k: v for k, v in m8.items() if k != "predictions"}, indent=2))
    log("STEP 8 complete.")

    # STEP 9
    log("\nSTEP 9: ARIA FULL PIPELINE")
    tier2_map = {int(x["index"]): x for x in preds7}
    tier3_map = {int(x["index"]): x for x in predictions8}
    final_pred = []
    for i, r in eval_df2.iterrows():
        if tier2_map[i]["escalated"]:
            decision = tier3_map[i]["weighted_decision"] if i in tier3_map else tier2_map[i]["decision"]
            source = "tier3" if i in tier3_map else "tier2_fallback"
        else:
            decision = tier2_map[i]["decision"]
            source = "tier2"
        final_pred.append({"index": int(i), "category": r["category"], "true_label": r["label"], "decision": decision, "source": source})
    dff = pd.DataFrame(final_pred)
    m9 = compute_metrics_from_preds(dff["true_label"].tolist(), dff["decision"].tolist(), dff["category"].tolist())
    m9["predictions"] = final_pred
    safe_json(OUTPUT_DIR / "results_aria_full.json", m9)
    log(json.dumps({k: v for k, v in m9.items() if k != "predictions"}, indent=2))
    log("STEP 9 complete.")

    # STEP 10
    log("\nSTEP 10: ABLATION TABLE")
    ablation = pd.DataFrame(
        [
            {
                "System": "BERT text-only",
                "Overall Accuracy": bert_text_metrics["overall_accuracy"],
                "Ambiguous Category Accuracy": bert_text_metrics["ambiguous_category_accuracy"],
                "Cost-Weighted Accuracy": bert_text_metrics["cost_weighted_accuracy"],
                "Notes": "Complaint text only no context",
            },
            {
                "System": "BERT text plus metadata",
                "Overall Accuracy": bert_meta_metrics["overall_accuracy"],
                "Ambiguous Category Accuracy": bert_meta_metrics["ambiguous_category_accuracy"],
                "Cost-Weighted Accuracy": bert_meta_metrics["cost_weighted_accuracy"],
                "Notes": "Text plus category tag",
            },
            {
                "System": "Groq direct routing",
                "Overall Accuracy": metrics6["overall_accuracy"],
                "Ambiguous Category Accuracy": metrics6["ambiguous_category_accuracy"],
                "Cost-Weighted Accuracy": metrics6["cost_weighted_accuracy"],
                "Notes": "Strong LLM no building context",
            },
            {
                "System": "ARIA Tier 2 only",
                "Overall Accuracy": m7["overall_accuracy"],
                "Ambiguous Category Accuracy": m7["ambiguous_category_accuracy"],
                "Cost-Weighted Accuracy": m7["cost_weighted_accuracy"],
                "Notes": "Context-augmented single agent",
            },
            {
                "System": "ARIA full pipeline",
                "Overall Accuracy": m9["overall_accuracy"],
                "Ambiguous Category Accuracy": m9["ambiguous_category_accuracy"],
                "Cost-Weighted Accuracy": m9["cost_weighted_accuracy"],
                "Notes": "Three-tier with cost-of-error arbiter",
            },
        ]
    )
    ablation.to_csv(OUTPUT_DIR / "ablation_table.csv", index=False)
    log(ablation.to_string(index=False))
    log("STEP 10 complete.")

    # STEP 11
    log("\nSTEP 11: PAPER-READY SUMMARY")
    changed_total = m8["cost_arbiter_changed_decisions"]
    changed_ok = m8["changed_decisions_matching_ground_truth"]
    changed_pct = (changed_ok / changed_total * 100) if changed_total else 0.0
    summary = []
    summary.append("DATASET")
    summary.append(f"Total complaints: {stats['total_complaints']}")
    summary.append(f"Mature communities: {df_mature['community_id'].nunique()}")
    summary.append(f"Date range: {stats['date_range']['earliest']} to {stats['date_range']['latest']}")
    summary.append(f"FM ownership: {stats['overall_split_percent']['FM']:.2f} percent")
    summary.append(f"Project ownership: {stats['overall_split_percent']['Project']:.2f} percent")
    summary.append(f"Ambiguous category splits: {json.dumps(stats['ambiguous_category_splits'])}")
    summary.append(f"Repeat complaint rate: {stats['repeat_complaint_rate_percent']:.2f} percent")
    summary.append(f"Flat-category combinations logged 5+ times: {stats['flat_category_combinations_5_or_more']}")
    summary.append("")
    summary.append("RESULTS TABLE")
    summary.append(ablation.to_string(index=False))
    summary.append("")
    summary.append("KEY FINDINGS")
    summary.append(
        f"Cost-of-error arbiter changed routing decision in {changed_total} out of {m8['sample_size']} Tier 3 complaints versus pure probability routing."
    )
    summary.append(
        f"Of those {changed_total} changed decisions, {changed_ok} matched ground truth ({changed_pct:.2f} percent)."
    )
    summary.append(
        f"ARIA full pipeline achieves {m9['cost_weighted_accuracy']*100:.2f} percent cost-weighted accuracy versus {bert_text_metrics['cost_weighted_accuracy']*100:.2f} percent for BERT text-only baseline."
    )
    summary.append(
        f"ARIA full pipeline achieves {m9['ambiguous_category_accuracy']*100:.2f} percent accuracy on ambiguous categories versus {bert_text_metrics['ambiguous_category_accuracy']*100:.2f} percent for BERT text-only."
    )

    summary_text = "\n".join(summary)
    with open(OUTPUT_DIR / "paper_summary.txt", "w", encoding="utf-8") as f:
        f.write(summary_text)
    log(summary_text)
    log("STEP 11 complete.")
    log("\nAll steps completed successfully.")


if __name__ == "__main__":
    main()
