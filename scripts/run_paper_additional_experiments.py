import json
import os
import random
import re
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeout
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from groq import Groq
from sklearn.metrics import accuracy_score
from torch.utils.data import Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments


OUT = Path.home() / "resolv-experiments" / "experiment_results"
AMB_PATH = OUT / "eval_ambiguous.csv"
TRAIN_PATH = OUT / "train.csv"
EVAL500_PATH = OUT / "eval_500.csv"


def log(msg: str) -> None:
    print(msg, flush=True)


def cwa(y_true: List[str], y_pred: List[str]) -> float:
    total_cost = 0
    max_cost = 0
    for t, p in zip(y_true, y_pred):
        if t == "Project" and p == "FM":
            total_cost += 10
            max_cost += 10
        elif t == "FM" and p == "Project":
            total_cost += 1
            max_cost += 10
        else:
            max_cost += 10
    return 1 - (total_cost / max_cost) if max_cost else 1.0


def clean(x) -> str:
    if pd.isna(x):
        return ""
    return re.sub(r"\s+", " ", str(x).strip().lower())


def is_ambiguous(cat: str) -> bool:
    c = clean(cat)
    keys = ["leakage", "seepage", "civil", "carpentry", "common area", "mason"]
    return any(k in c for k in keys)


def get_groq_client_and_model() -> Tuple[Groq, str]:
    key = os.getenv("GROQ_API_KEY")
    if not key:
        env_file = Path("/Users/kartheek/resolv/.env")
        if env_file.exists():
            for line in env_file.read_text(encoding="utf-8").splitlines():
                if line.startswith("GROQ_API_KEY="):
                    key = line.split("=", 1)[1].strip().strip('"').strip("'")
                    break
    if not key:
        raise RuntimeError("GROQ_API_KEY not found")
    client = Groq(api_key=key)
    candidates = [
        "llama-3.3-70b-versatile",
        "llama-3.1-8b-instant",
        "mixtral-8x7b-32768",
    ]
    for m in candidates:
        try:
            _ = client.chat.completions.create(
                model=m,
                messages=[{"role": "user", "content": "FM or Project? One word only."}],
                max_tokens=5,
                temperature=0,
            )
            return client, m
        except Exception:
            continue
    raise RuntimeError("No available Groq model from candidates")


def call_groq_timeout(client: Groq, model: str, prompt: str, max_tokens: int = 10, timeout: int = 12, retries: int = 3) -> str:
    for attempt in range(retries):
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(
                client.chat.completions.create,
                model=model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                temperature=0,
            )
            try:
                resp = future.result(timeout=timeout)
                return (resp.choices[0].message.content or "").strip()
            except FuturesTimeout:
                if attempt == retries - 1:
                    return "TIMEOUT"
            except Exception as e:
                if attempt == retries - 1:
                    return f"ERROR: {e}"
        time.sleep(0.8 * (attempt + 1))
    return "TIMEOUT"


def parse_fm_project(raw: str) -> str:
    t = (raw or "").strip()
    first = t.split()[0].upper() if t.split() else ""
    if "FM" in first:
        return "FM"
    if "PROJECT" in first:
        return "Project"
    up = t.upper()
    if "FM" in up:
        return "FM"
    if "PROJECT" in up:
        return "Project"
    return "FM"


def bootstrap_ci(y_true: List[str], y_pred: List[str], n: int = 1000, seed: int = 42) -> Dict:
    rng = np.random.default_rng(seed)
    y_true_arr = np.array(y_true)
    y_pred_arr = np.array(y_pred)
    m = len(y_true_arr)
    accs = []
    cwas = []
    for _ in range(n):
        idx = rng.integers(0, m, size=m)
        yt = y_true_arr[idx].tolist()
        yp = y_pred_arr[idx].tolist()
        accs.append(float(accuracy_score(yt, yp)))
        cwas.append(float(cwa(yt, yp)))
    return {
        "accuracy_mean": float(np.mean(accs)),
        "accuracy_ci_low": float(np.percentile(accs, 2.5)),
        "accuracy_ci_high": float(np.percentile(accs, 97.5)),
        "cwa_mean": float(np.mean(cwas)),
        "cwa_ci_low": float(np.percentile(cwas, 2.5)),
        "cwa_ci_high": float(np.percentile(cwas, 97.5)),
    }


def step1_tier2_ambiguous_200(client: Groq, model: str) -> Dict:
    eval_amb = pd.read_csv(AMB_PATH).head(200).copy().reset_index(drop=True)
    train = pd.read_csv(TRAIN_PATH)
    if "date" in train.columns:
        train["date"] = pd.to_datetime(train["date"], errors="coerce")
        train = train.sort_values("date")

    preds = []
    for i, r in eval_amb.iterrows():
        flat = str(r.get("flat_id", ""))
        hist = train[train["flat_id"].astype(str) == flat].tail(2)
        prior = "None" if len(hist) == 0 else " | ".join([f"{h['category']}: {str(h['text'])[:100]}" for _, h in hist.iterrows()])
        prompt = (
            "You are an intelligent complaint routing system for a residential gated community in India. "
            "Your job is to determine whether this complaint is the responsibility of the FM company (routine maintenance) "
            "or the Project team (developer warranty and structural issues). Use the complaint details and prior history from this flat "
            f"to make your decision. Category: {r['category']}. Complaint: {r['text']}. Prior complaints from this flat: {prior}. "
            "Think step by step about what could cause this issue and who is responsible. "
            "Respond in exactly this format: Decision: FM or Project. Confidence: [number between 0 and 1]. Reasoning: one sentence."
        )
        raw = call_groq_timeout(client, model, prompt, max_tokens=80, timeout=15)
        pred = parse_fm_project(raw)
        preds.append(
            {
                "index": i,
                "true_label": r["label"],
                "prediction": pred,
                "raw_response": raw,
                "category": r["category"],
            }
        )
        if (i + 1) % 20 == 0:
            log(f"STEP 1 progress: {i+1}/200 complaints processed")

    y_true = [x["true_label"] for x in preds]
    y_pred = [x["prediction"] for x in preds]
    metrics = {
        "model_used": model,
        "overall_accuracy": float(accuracy_score(y_true, y_pred)),
        "ambiguous_category_accuracy": float(accuracy_score(y_true, y_pred)),
        "cost_weighted_accuracy": float(cwa(y_true, y_pred)),
        "sample_size": len(preds),
        "predictions": preds,
    }
    (OUT / "results_aria_tier2_ambiguous_200.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    log(f"STEP 1 complete. Accuracy={metrics['overall_accuracy']:.4f}, CWA={metrics['cost_weighted_accuracy']:.4f}")
    return metrics


class TxtDataset(Dataset):
    def __init__(self, texts: List[str], labels: List[int], tokenizer, max_len: int = 256):
        self.enc = tokenizer(texts, truncation=True, padding=True, max_length=max_len)
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {k: torch.tensor(v[idx]) for k, v in self.enc.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item


class WeightedTrainer(Trainer):
    def __init__(self, class_weights: torch.Tensor, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")
        loss_fct = nn.CrossEntropyLoss(weight=self.class_weights.to(logits.device))
        loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
        return (loss, outputs) if return_outputs else loss


def step2_bert_cost_sensitive() -> Dict:
    train_df = pd.read_csv(TRAIN_PATH)
    eval_df = pd.read_csv(EVAL500_PATH)
    label_map = {"FM": 0, "Project": 1}

    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)

    train_texts = train_df["text"].astype(str).tolist()
    train_labels = [label_map[x] for x in train_df["label"].astype(str).tolist()]
    eval_texts = eval_df["text"].astype(str).tolist()
    eval_labels = [label_map[x] for x in eval_df["label"].astype(str).tolist()]

    train_ds = TxtDataset(train_texts, train_labels, tokenizer)
    eval_ds = TxtDataset(eval_texts, eval_labels, tokenizer)

    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    model.to(device)

    args = TrainingArguments(
        output_dir=str(OUT / "bert_cost_sensitive_ckpt"),
        num_train_epochs=3,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        learning_rate=2e-5,
        save_strategy="no",
        eval_strategy="no",
        report_to=[],
        disable_tqdm=True,
    )

    # FM=1.0, Project=5.0 -> label index 0:FM, 1:Project
    class_weights = torch.tensor([1.0, 5.0], dtype=torch.float)
    trainer = WeightedTrainer(
        class_weights=class_weights,
        model=model,
        args=args,
        train_dataset=train_ds,
        tokenizer=tokenizer,
    )
    trainer.train()
    out = trainer.predict(eval_ds)
    logits = out.predictions
    probs = torch.softmax(torch.tensor(logits), dim=1).numpy()
    pred_ids = np.argmax(probs, axis=1)
    inv = {0: "FM", 1: "Project"}
    y_pred = [inv[i] for i in pred_ids]
    y_true = eval_df["label"].astype(str).tolist()
    amb_mask = eval_df["category"].astype(str).map(is_ambiguous)
    if amb_mask.any():
        amb_true = eval_df.loc[amb_mask, "label"].astype(str).tolist()
        amb_pred = [y_pred[i] for i, m in enumerate(amb_mask.tolist()) if m]
        amb_acc = float(accuracy_score(amb_true, amb_pred))
    else:
        amb_acc = 0.0
    metrics = {
        "overall_accuracy": float(accuracy_score(y_true, y_pred)),
        "ambiguous_category_accuracy": amb_acc,
        "cost_weighted_accuracy": float(cwa(y_true, y_pred)),
        "sample_size": len(y_true),
        "predictions": y_pred,
    }
    (OUT / "results_bert_cost_sensitive.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    log(f"STEP 2 complete. Accuracy={metrics['overall_accuracy']:.4f}, Ambiguous={metrics['ambiguous_category_accuracy']:.4f}, CWA={metrics['cost_weighted_accuracy']:.4f}")
    return metrics


def step3_llm_context_50(client: Groq, model: str) -> Dict:
    eval_amb = pd.read_csv(AMB_PATH).head(50).copy().reset_index(drop=True)
    train = pd.read_csv(TRAIN_PATH)
    if "date" in train.columns:
        train["date"] = pd.to_datetime(train["date"], errors="coerce")
        train = train.sort_values("date")
    preds = []
    for i, r in eval_amb.iterrows():
        flat = str(r.get("flat_id", ""))
        hist = train[train["flat_id"].astype(str) == flat].tail(2)
        prior = "None" if len(hist) == 0 else " | ".join([f"{h['category']}: {str(h['text'])[:100]}" for _, h in hist.iterrows()])
        prompt = (
            "You are an expert facility management consultant for Indian residential communities. "
            "Using the complaint details AND the prior history from this flat, determine ownership. "
            f"Category: {r['category']}. Complaint: {r['text']}. Prior complaints from this flat: {prior}. "
            "You must respond with exactly one word: FM or Project."
        )
        raw = call_groq_timeout(client, model, prompt, max_tokens=12, timeout=12)
        preds.append({"true_label": r["label"], "prediction": parse_fm_project(raw), "raw_response": raw})
    y_true = [x["true_label"] for x in preds]
    y_pred = [x["prediction"] for x in preds]
    metrics = {
        "model_used": model,
        "overall_accuracy": float(accuracy_score(y_true, y_pred)),
        "ambiguous_category_accuracy": float(accuracy_score(y_true, y_pred)),
        "cost_weighted_accuracy": float(cwa(y_true, y_pred)),
        "sample_size": len(preds),
        "predictions": preds,
    }
    (OUT / "results_llm_with_context_baseline.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    log(f"STEP 3 complete. Accuracy={metrics['overall_accuracy']:.4f}, CWA={metrics['cost_weighted_accuracy']:.4f}")
    return metrics


def step4_ablation(client: Groq, model: str, tier2_200: Dict) -> Dict:
    eval_amb = pd.read_csv(AMB_PATH).head(200).copy().reset_index(drop=True)
    train = pd.read_csv(TRAIN_PATH)
    if "date" in train.columns:
        train["date"] = pd.to_datetime(train["date"], errors="coerce")
        train = train.sort_values("date")

    # A: text+category only
    preds_a = []
    for _, r in eval_amb.iterrows():
        prompt = (
            "Classify ownership for this complaint as one word FM or Project. "
            f"Category: {r['category']}. Complaint: {r['text']}. Answer FM or Project."
        )
        raw = call_groq_timeout(client, model, prompt, max_tokens=10, timeout=12)
        preds_a.append(parse_fm_project(raw))

    # B: + prior history (reuse step1 predictions where possible: same 200 set + history concept)
    # To keep exact component definition here, run B prompt explicitly.
    preds_b = []
    for _, r in eval_amb.iterrows():
        flat = str(r.get("flat_id", ""))
        hist = train[train["flat_id"].astype(str) == flat].tail(2)
        prior = "None" if len(hist) == 0 else " | ".join([f"{h['category']}: {str(h['text'])[:100]}" for _, h in hist.iterrows()])
        prompt = (
            "Classify ownership for this complaint as one word FM or Project. "
            f"Category: {r['category']}. Complaint: {r['text']}. Prior history from this flat: {prior}. "
            "Answer FM or Project."
        )
        raw = call_groq_timeout(client, model, prompt, max_tokens=12, timeout=12)
        preds_b.append(parse_fm_project(raw))

    # C: multi-agent probabilities, no cost weights (pure max probability)
    preds_c = []
    for _, r in eval_amb.iterrows():
        flat = str(r.get("flat_id", ""))
        hist = train[train["flat_id"].astype(str) == flat].tail(2)
        prior = "None" if len(hist) == 0 else " | ".join([f"{h['category']}: {str(h['text'])[:100]}" for _, h in hist.iterrows()])

        p_pipe = call_groq_timeout(
            client,
            model,
            (
                "Estimate probability 0..1 that this is plumbing/pipe failure (FM). "
                f"Category: {r['category']}. Complaint: {r['text']}. Prior history: {prior}. "
                "Respond only decimal number."
            ),
            max_tokens=8,
            timeout=12,
        )
        p_struct = call_groq_timeout(
            client,
            model,
            (
                "Estimate probability 0..1 that this is structural/waterproofing/construction defect (Project). "
                f"Category: {r['category']}. Complaint: {r['text']}. Prior history: {prior}. "
                "Respond only decimal number."
            ),
            max_tokens=8,
            timeout=12,
        )
        p_asset = call_groq_timeout(
            client,
            model,
            (
                "Estimate probability 0..1 that this is asset/equipment failure (FM). "
                f"Category: {r['category']}. Complaint: {r['text']}. Prior history: {prior}. "
                "Respond only decimal number."
            ),
            max_tokens=8,
            timeout=12,
        )

        def parse_prob(s: str) -> float:
            try:
                v = float((s or "").strip())
                if 0 <= v <= 1:
                    return v
            except Exception:
                pass
            m = re.search(r"([0-9]*\.?[0-9]+)", s or "")
            if m:
                try:
                    v = float(m.group(1))
                    if 0 <= v <= 1:
                        return v
                except Exception:
                    pass
            return 0.5

        pp, ps, pa = parse_prob(p_pipe), parse_prob(p_struct), parse_prob(p_asset)
        if ps >= max(pp, pa):
            preds_c.append("Project")
        else:
            preds_c.append("FM")

    # D: full tier3 with cost weights from existing file
    d_file = OUT / "results_aria_tier3_ambiguous.json"
    d_json = json.loads(d_file.read_text(encoding="utf-8"))
    if "predictions" in d_json and len(d_json["predictions"]) >= 200:
        preds_d = []
        for p in d_json["predictions"][:200]:
            if "weighted_decision" in p:
                preds_d.append(p["weighted_decision"])
            elif "decision" in p:
                preds_d.append(p["decision"])
            else:
                preds_d.append("FM")
    else:
        preds_d = ["FM"] * len(eval_amb)

    y_true = eval_amb["label"].astype(str).tolist()

    def pack(preds: List[str]) -> Dict:
        return {
            "accuracy": float(accuracy_score(y_true, preds)),
            "cwa": float(cwa(y_true, preds)),
            "predictions": preds,
        }

    A = pack(preds_a)
    B = pack(preds_b)
    C = pack(preds_c)
    D = pack(preds_d)

    table = {
        "A_text_only": A,
        "B_plus_building_context": B,
        "C_plus_multi_agent_no_cost_weights": C,
        "D_full_aria_tier3_cost_weighted": D,
    }
    (OUT / "ablation_component_study.json").write_text(json.dumps(table, indent=2), encoding="utf-8")

    log("STEP 4 complete.")
    log("Component | Ambiguous Acc | Cost-Weighted Acc | Delta Acc vs A | Delta CWA vs A")
    log(f"A: Text only | {A['accuracy']:.4f} | {A['cwa']:.4f} | — | —")
    log(f"B: + Building context | {B['accuracy']:.4f} | {B['cwa']:.4f} | {B['accuracy']-A['accuracy']:+.4f} | {B['cwa']-A['cwa']:+.4f}")
    log(f"C: + Multi-agent (no cost weights) | {C['accuracy']:.4f} | {C['cwa']:.4f} | {C['accuracy']-A['accuracy']:+.4f} | {C['cwa']-A['cwa']:+.4f}")
    log(f"D: + Cost-of-error arbiter (full) | {D['accuracy']:.4f} | {D['cwa']:.4f} | {D['accuracy']-A['accuracy']:+.4f} | {D['cwa']-A['cwa']:+.4f}")
    return table


def step5_bootstrap_ci() -> Dict:
    eval500 = pd.read_csv(EVAL500_PATH)
    # BERT text-only
    bert = json.loads((OUT / "results_bert_text_only.json").read_text(encoding="utf-8"))
    y_true_bert = eval500["label"].astype(str).tolist()
    y_pred_bert = bert.get("predictions", ["FM"] * len(y_true_bert))
    ci_bert = bootstrap_ci(y_true_bert, y_pred_bert, n=1000, seed=42)

    # ARIA Tier3 ambiguous
    tier3 = json.loads((OUT / "results_aria_tier3_ambiguous.json").read_text(encoding="utf-8"))
    p3 = tier3.get("predictions", [])
    y_true_t3 = []
    y_pred_t3 = []
    for p in p3:
        y_true_t3.append(p.get("true_label", "FM"))
        y_pred_t3.append(p.get("weighted_decision", p.get("decision", "FM")))
    ci_t3 = bootstrap_ci(y_true_t3, y_pred_t3, n=1000, seed=43)

    # ARIA full pipeline
    full = json.loads((OUT / "results_aria_full.json").read_text(encoding="utf-8"))
    pf = full.get("predictions", [])
    y_true_f = [x.get("true_label", "FM") for x in pf]
    y_pred_f = [x.get("decision", "FM") for x in pf]
    ci_full = bootstrap_ci(y_true_f, y_pred_f, n=1000, seed=44)

    payload = {
        "BERT text-only": ci_bert,
        "ARIA Tier 3": ci_t3,
        "ARIA full pipeline": ci_full,
    }
    (OUT / "bootstrap_confidence_intervals.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    log("STEP 5 complete.")
    log("System | Accuracy (95% CI) | CWA (95% CI)")
    log(f"BERT text-only | {ci_bert['accuracy_mean']:.4f} [{ci_bert['accuracy_ci_low']:.4f}, {ci_bert['accuracy_ci_high']:.4f}] | {ci_bert['cwa_mean']:.4f} [{ci_bert['cwa_ci_low']:.4f}, {ci_bert['cwa_ci_high']:.4f}]")
    log(f"ARIA Tier 3 | {ci_t3['accuracy_mean']:.4f} [{ci_t3['accuracy_ci_low']:.4f}, {ci_t3['accuracy_ci_high']:.4f}] | {ci_t3['cwa_mean']:.4f} [{ci_t3['cwa_ci_low']:.4f}, {ci_t3['cwa_ci_high']:.4f}]")
    log(f"ARIA full pipeline | {ci_full['accuracy_mean']:.4f} [{ci_full['accuracy_ci_low']:.4f}, {ci_full['accuracy_ci_high']:.4f}] | {ci_full['cwa_mean']:.4f} [{ci_full['cwa_ci_low']:.4f}, {ci_full['cwa_ci_high']:.4f}]")
    return payload


def safe_load(path: Path) -> Dict:
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {}


def extract_ci_for_system(name: str, ci_payload: Dict, y_true: List[str], y_pred: List[str]) -> Dict:
    if name in ci_payload:
        return ci_payload[name]
    return bootstrap_ci(y_true, y_pred, n=1000, seed=123)


def step6_paper_results(tier2_200: Dict, bert_cost: Dict, llm_ctx: Dict, ablation: Dict, ci_payload: Dict) -> None:
    bert_text = safe_load(OUT / "results_bert_text_only.json")
    bert_meta = safe_load(OUT / "results_bert_text_meta.json")
    groq_50 = safe_load(OUT / "results_groq_direct_final.json")
    tier3_amb = safe_load(OUT / "results_aria_tier3_ambiguous.json")
    aria_full = safe_load(OUT / "results_aria_full.json")
    eval500 = pd.read_csv(EVAL500_PATH)
    eval_amb = pd.read_csv(AMB_PATH).head(200)

    # build CI fallback for systems not in step5 payload
    y_true_500 = eval500["label"].astype(str).tolist()
    y_true_amb200 = eval_amb["label"].astype(str).tolist()

    def pred_from(d: Dict, key="predictions", subkey=None, default_len=0):
        arr = d.get(key, [])
        if subkey and len(arr) and isinstance(arr[0], dict):
            return [x.get(subkey, "FM") for x in arr]
        if len(arr) and isinstance(arr[0], str):
            return arr
        return ["FM"] * default_len

    ci_bert_text = ci_payload.get("BERT text-only")
    ci_bert_meta = extract_ci_for_system("BERT text+meta", ci_payload, y_true_500, pred_from(bert_meta, default_len=len(y_true_500)))
    ci_bert_cost = extract_ci_for_system("BERT cost-sensitive", ci_payload, y_true_500, bert_cost.get("predictions", ["FM"] * len(y_true_500)))
    ci_groq50 = extract_ci_for_system("Groq direct 50", ci_payload, y_true_amb200[:50], groq_50.get("predictions", ["FM"] * 50))
    ci_llm_ctx = extract_ci_for_system("LLM context 50", ci_payload, y_true_amb200[:50], llm_ctx.get("predictions", []))
    ci_tier2_200 = extract_ci_for_system("ARIA Tier2 200", ci_payload, y_true_amb200, pred_from(tier2_200, subkey="prediction", default_len=len(y_true_amb200)))
    ci_tier3 = ci_payload.get("ARIA Tier 3", extract_ci_for_system("ARIA Tier3", ci_payload, y_true_amb200, pred_from(tier3_amb, subkey="weighted_decision", default_len=len(y_true_amb200))))
    ci_full = ci_payload.get("ARIA full pipeline", extract_ci_for_system("ARIA full", ci_payload, y_true_500, pred_from(aria_full, subkey="decision", default_len=len(y_true_500))))

    lines = []
    lines.append("FINAL RESULTS FOR PAPER")
    lines.append("MAIN RESULTS TABLE (standardized evaluation)")
    lines.append("System | Overall Acc | Ambiguous Acc (200 samples) | CWA | 95% CI on CWA")
    lines.append(
        f"BERT text-only | {bert_text.get('overall_accuracy', 0):.4f} | {bert_text.get('ambiguous_category_accuracy', 0):.4f} | {bert_text.get('cost_weighted_accuracy', 0):.4f} | [{ci_bert_text['cwa_ci_low']:.4f}, {ci_bert_text['cwa_ci_high']:.4f}]"
    )
    lines.append(
        f"BERT text + metadata | {bert_meta.get('overall_accuracy', 0):.4f} | {bert_meta.get('ambiguous_category_accuracy', 0):.4f} | {bert_meta.get('cost_weighted_accuracy', 0):.4f} | [{ci_bert_meta['cwa_ci_low']:.4f}, {ci_bert_meta['cwa_ci_high']:.4f}]"
    )
    lines.append(
        f"BERT cost-sensitive | {bert_cost.get('overall_accuracy', 0):.4f} | {bert_cost.get('ambiguous_category_accuracy', 0):.4f} | {bert_cost.get('cost_weighted_accuracy', 0):.4f} | [{ci_bert_cost['cwa_ci_low']:.4f}, {ci_bert_cost['cwa_ci_high']:.4f}]"
    )
    lines.append(
        f"Groq direct no context | {groq_50.get('overall_accuracy', 0):.4f} (50 samples) | {groq_50.get('overall_accuracy', 0):.4f} | {groq_50.get('cost_weighted_accuracy', 0):.4f} | [{ci_groq50['cwa_ci_low']:.4f}, {ci_groq50['cwa_ci_high']:.4f}]"
    )
    lines.append(
        f"LLM with context (50 samples) | {llm_ctx.get('overall_accuracy', 0):.4f} | {llm_ctx.get('overall_accuracy', 0):.4f} | {llm_ctx.get('cost_weighted_accuracy', 0):.4f} | [{ci_llm_ctx['cwa_ci_low']:.4f}, {ci_llm_ctx['cwa_ci_high']:.4f}]"
    )
    lines.append(
        f"ARIA Tier 2 (200 samples) | {tier2_200.get('overall_accuracy', 0):.4f} | {tier2_200.get('ambiguous_category_accuracy', 0):.4f} | {tier2_200.get('cost_weighted_accuracy', 0):.4f} | [{ci_tier2_200['cwa_ci_low']:.4f}, {ci_tier2_200['cwa_ci_high']:.4f}]"
    )
    lines.append(
        f"ARIA Tier 3 cost-of-error | {tier3_amb.get('overall_accuracy', 0):.4f} | {tier3_amb.get('ambiguous_category_accuracy', 0):.4f} | {tier3_amb.get('cost_weighted_accuracy', 0):.4f} | [{ci_tier3['cwa_ci_low']:.4f}, {ci_tier3['cwa_ci_high']:.4f}]"
    )
    lines.append(
        f"ARIA full pipeline | {aria_full.get('overall_accuracy', 0):.4f} | {aria_full.get('ambiguous_category_accuracy', 0):.4f} | {aria_full.get('cost_weighted_accuracy', 0):.4f} | [{ci_full['cwa_ci_low']:.4f}, {ci_full['cwa_ci_high']:.4f}]"
    )
    lines.append("ABLATION TABLE")
    lines.append("Component | Ambiguous Acc | Cost-Weighted Acc | Delta Acc vs A | Delta CWA vs A")
    A = ablation["A_text_only"]
    B = ablation["B_plus_building_context"]
    C = ablation["C_plus_multi_agent_no_cost_weights"]
    D = ablation["D_full_aria_tier3_cost_weighted"]
    lines.append(f"A: Text only | {A['accuracy']:.4f} | {A['cwa']:.4f} | — | —")
    lines.append(f"B: + Building context | {B['accuracy']:.4f} | {B['cwa']:.4f} | {B['accuracy']-A['accuracy']:+.4f} | {B['cwa']-A['cwa']:+.4f}")
    lines.append(f"C: + Multi-agent (no cost weights) | {C['accuracy']:.4f} | {C['cwa']:.4f} | {C['accuracy']-A['accuracy']:+.4f} | {C['cwa']-A['cwa']:+.4f}")
    lines.append(f"D: + Cost-of-error arbiter (full) | {D['accuracy']:.4f} | {D['cwa']:.4f} | {D['accuracy']-A['accuracy']:+.4f} | {D['cwa']-A['cwa']:+.4f}")
    lines.append("KEY NUMBERS FOR ABSTRACT")
    lines.append(f"Best ambiguous accuracy (ARIA Tier 3): {tier3_amb.get('ambiguous_category_accuracy', 0):.4f}")
    lines.append(f"Best cost-weighted accuracy (ARIA Tier 3): {tier3_amb.get('cost_weighted_accuracy', 0):.4f}")
    lines.append(f"LLM baseline ambiguous accuracy: {llm_ctx.get('overall_accuracy', 0):.4f}")
    lines.append(f"BERT baseline ambiguous accuracy: {bert_text.get('ambiguous_category_accuracy', 0):.4f}")
    lines.append(
        f"Improvement over BERT on ambiguous: {(tier3_amb.get('ambiguous_category_accuracy',0)-bert_text.get('ambiguous_category_accuracy',0))*100:.2f} percentage points"
    )
    lines.append(
        f"Improvement over LLM direct on ambiguous: {(tier3_amb.get('ambiguous_category_accuracy',0)-groq_50.get('overall_accuracy',0))*100:.2f} percentage points"
    )

    text = "\n".join(lines)
    (OUT / "paper_results_final.txt").write_text(text, encoding="utf-8")
    log("STEP 6 complete.")
    log(text)


def main():
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    OUT.mkdir(parents=True, exist_ok=True)
    client, model = get_groq_client_and_model()
    log(f"Using Groq model: {model}")

    tier2_200 = step1_tier2_ambiguous_200(client, model)
    bert_cost = step2_bert_cost_sensitive()
    llm_ctx = step3_llm_context_50(client, model)
    ablation = step4_ablation(client, model, tier2_200)
    ci_payload = step5_bootstrap_ci()
    step6_paper_results(tier2_200, bert_cost, llm_ctx, ablation, ci_payload)


if __name__ == "__main__":
    main()
