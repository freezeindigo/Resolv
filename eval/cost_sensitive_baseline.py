"""
Cost-sensitive logistic regression baselines for ARIA paper comparison.

Usage:
    python3 eval/cost_sensitive_baseline.py
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, List

import numpy as np
import psycopg2
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder


SEED = 42
DB_NAME = "resolv"
OUT_PATH = Path("eval/results/cost_sensitive_baseline_results.csv")

# User-requested ambiguous categories
RAW_CATEGORIES = [
    "Leakage",
    "Seepage",
    "Mason",
    "Civil",
    "Fitout",
    "Painter",
    "Civil Work",
    "Carpentry",
    "Common Area",
    "Plumbing",
    "Amenities",
    "Electrical",
]


def fetch_rows() -> List[dict]:
    category_set = {c.lower() for c in RAW_CATEGORIES}
    # Handle common data spelling in this DB.
    category_set.add("carpentary")

    conn = psycopg2.connect(dbname=DB_NAME)
    cur = conn.cursor()
    cur.execute(
        """
        SELECT complaint_title, category, priority, issue_type
        FROM complaints
        WHERE complaint_title IS NOT NULL
          AND issue_type IS NOT NULL
          AND category IS NOT NULL
        """
    )
    out = []
    for title, category, priority, issue_type in cur.fetchall():
        cat = (category or "").strip()
        if cat.lower() not in category_set:
            continue
        own = (issue_type or "").strip().lower()
        if "project" in own:
            y = 1
        elif "fm" in own or "facility" in own:
            y = 0
        else:
            continue
        out.append(
            {
                "title": str(title),
                "category": cat,
                "priority": (str(priority) if priority is not None else "Unknown"),
                "label": y,  # FM=0, Project=1
            }
        )
    cur.close()
    conn.close()
    return out


def build_features(train_rows: List[dict], test_rows: List[dict]):
    tfidf = TfidfVectorizer(max_features=500)
    ohe = OneHotEncoder(handle_unknown="ignore")

    train_titles = [r["title"] for r in train_rows]
    test_titles = [r["title"] for r in test_rows]
    X_train_text = tfidf.fit_transform(train_titles)
    X_test_text = tfidf.transform(test_titles)

    train_meta = np.array([[r["category"], r["priority"]] for r in train_rows], dtype=object)
    test_meta = np.array([[r["category"], r["priority"]] for r in test_rows], dtype=object)
    X_train_meta = ohe.fit_transform(train_meta)
    X_test_meta = ohe.transform(test_meta)

    X_train = hstack([X_train_text, X_train_meta], format="csr")
    X_test = hstack([X_test_text, X_test_meta], format="csr")
    y_train = np.array([r["label"] for r in train_rows], dtype=int)
    y_test = np.array([r["label"] for r in test_rows], dtype=int)
    return X_train, X_test, y_train, y_test


def evaluate(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    overall = accuracy_score(y_true, y_pred)
    fm_mask = y_true == 0
    proj_mask = y_true == 1
    fm_acc = float((y_pred[fm_mask] == 0).mean()) if fm_mask.any() else 0.0
    proj_acc = float((y_pred[proj_mask] == 1).mean()) if proj_mask.any() else 0.0

    high_cost_errors = int(((y_true == 1) & (y_pred == 0)).sum())  # Project->FM (cost 10)
    low_cost_errors = int(((y_true == 0) & (y_pred == 1)).sum())  # FM->Project (cost 1)
    weighted_cost = high_cost_errors * 10 + low_cost_errors * 1

    return {
        "overall": float(overall),
        "fm_acc": fm_acc,
        "proj_acc": proj_acc,
        "high_cost": high_cost_errors,
        "low_cost": low_cost_errors,
        "weighted_cost": int(weighted_cost),
    }


def pct(v: float) -> str:
    return f"{v*100:.1f}%"


def delta_vs_gpt(weighted_cost: int, gpt_weighted_cost: int) -> str:
    change = ((weighted_cost - gpt_weighted_cost) / gpt_weighted_cost) * 100.0
    return f"{change:+.1f}%"


def main():
    rows = fetch_rows()
    if not rows:
        raise RuntimeError("No rows found for requested categories.")

    train_rows, test_rows = train_test_split(
        rows,
        test_size=0.30,
        random_state=SEED,
        stratify=[r["label"] for r in rows],
    )
    X_train, X_test, y_train, y_test = build_features(train_rows, test_rows)

    # 1) LR-Balanced
    lr_bal = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=SEED)
    lr_bal.fit(X_train, y_train)
    pred_bal = lr_bal.predict(X_test)
    m_bal = evaluate(y_test, pred_bal)

    # 2) LR-CostWeighted
    lr_cost = LogisticRegression(class_weight={0: 1, 1: 10}, max_iter=1000, random_state=SEED)
    lr_cost.fit(X_train, y_train)
    pred_cost = lr_cost.predict(X_test)
    m_cost = evaluate(y_test, pred_cost)

    # 3) LR-ThresholdShift
    # Train balanced, then threshold at 0.1 for Project (class 1)
    lr_thr = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=SEED)
    lr_thr.fit(X_train, y_train)
    p_project = lr_thr.predict_proba(X_test)[:, 1]
    pred_thr = (p_project >= 0.1).astype(int)
    m_thr = evaluate(y_test, pred_thr)

    # Fixed comparator rows requested by user.
    gpt_row = {
        "system": "GPT-4o (text-only)",
        "overall": 0.590,
        "fm_acc": 0.916,
        "proj_acc": 0.123,
        "high_cost": 107,
        "low_cost": 15,
        "weighted_cost": 1085,
        "vs_gpt": "baseline",
    }
    aria_row = {
        "system": "ARIA-Full",
        "overall": 0.590,
        "fm_acc": 0.725,
        "proj_acc": 0.393,
        "high_cost": 74,
        "low_cost": 49,
        "weighted_cost": 789,
        "vs_gpt": "-27.3%",
    }

    result_rows = [
        gpt_row,
        {
            "system": "LR-Balanced",
            **m_bal,
            "vs_gpt": delta_vs_gpt(m_bal["weighted_cost"], gpt_row["weighted_cost"]),
        },
        {
            "system": "LR-CostWeighted",
            **m_cost,
            "vs_gpt": delta_vs_gpt(m_cost["weighted_cost"], gpt_row["weighted_cost"]),
        },
        {
            "system": "LR-ThresholdShift",
            **m_thr,
            "vs_gpt": delta_vs_gpt(m_thr["weighted_cost"], gpt_row["weighted_cost"]),
        },
        aria_row,
    ]

    # Print in requested markdown style.
    print("| System              | Overall | FM Acc | Proj Acc | High-cost | Low-cost | Weighted Cost | vs GPT-4o |")
    print("|---------------------|---------|--------|----------|-----------|----------|---------------|-----------|")
    for r in result_rows:
        print(
            f"| {r['system']:<19} | "
            f"{pct(r['overall']):>7} | "
            f"{pct(r['fm_acc']):>6} | "
            f"{pct(r['proj_acc']):>8} | "
            f"{r['high_cost']:>9} | "
            f"{r['low_cost']:>8} | "
            f"{r['weighted_cost']:>13} | "
            f"{r['vs_gpt']:>9} |"
        )

    # Save CSV results
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "system",
                "overall",
                "fm_acc",
                "proj_acc",
                "high_cost",
                "low_cost",
                "weighted_cost",
                "vs_gpt",
            ],
        )
        writer.writeheader()
        for r in result_rows:
            writer.writerow(r)

    print(f"\nSaved results to: {OUT_PATH}")
    print(f"Rows used: {len(rows)} | Train: {len(train_rows)} | Test: {len(test_rows)} | Seed: {SEED}")


if __name__ == "__main__":
    main()
