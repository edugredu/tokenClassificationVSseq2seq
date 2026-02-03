#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import os
import re
import time
from difflib import SequenceMatcher
import unicodedata
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch
from datasets import load_dataset
from peft import PeftModel
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, StoppingCriteria, StoppingCriteriaList

entity_types = ["PROTEINAS", "NORMALIZABLES", "NO_NORMALIZABLES", "UNCLEAR"]


def ensure_chat_template(tokenizer, model_id: str, hf_token: str = ""):
    if getattr(tokenizer, "chat_template", None):
        return

    model_id_lower = model_id.lower()
    fallback_id = None
    if ("llama2" in model_id_lower or "llama-2-7b" in model_id_lower) and "chat" not in model_id_lower:
        fallback_id = "meta-llama/Llama-2-7b-chat-hf"
    elif ("llama3_2" in model_id_lower or "llama-3.2-3b" in model_id_lower) and "instruct" not in model_id_lower:
        fallback_id = "meta-llama/Llama-3.2-3B-Instruct"
    elif ("llama3_1" in model_id_lower or "llama-3.1-8b" in model_id_lower) and "instruct" not in model_id_lower:
        fallback_id = "meta-llama/Meta-Llama-3.1-8B-Instruct"
    elif ("llama3" in model_id_lower or "llama-3-8b" in model_id_lower) and "instruct" not in model_id_lower:
        fallback_id = "meta-llama/Meta-Llama-3-8B-Instruct"

    if not fallback_id:
        print("Warning: tokenizer.chat_template missing and no fallback found.")
        return

    try:
        fallback_tok = AutoTokenizer.from_pretrained(
            fallback_id,
            use_fast=True,
            token=hf_token or None,
        )
    except Exception as exc:
        print(f"Warning: failed to load fallback tokenizer {fallback_id}: {exc}")
        return

    if getattr(fallback_tok, "chat_template", None):
        tokenizer.chat_template = fallback_tok.chat_template
        print(f"Applied chat_template from {fallback_id}")
    else:
        print(f"Warning: fallback tokenizer {fallback_id} has no chat_template")


class JSONStoppingCriteria(StoppingCriteria):
    """Stop once a JSON array appears to be complete (only on generated tokens)."""

    def __init__(self, tokenizer, max_entities: int = 50):
        self.tokenizer = tokenizer
        self.max_entities = max_entities
        self.start_length = None

    def __call__(self, input_ids, scores, **kwargs):
        if self.start_length is None:
            self.start_length = input_ids.shape[1]
            return False
        if input_ids.shape[1] <= self.start_length:
            return False

        generated = input_ids[0][self.start_length :]
        if generated.shape[0] < 5:
            return False

        last_tokens = generated[-30:]
        try:
            text = self.tokenizer.decode(last_tokens, skip_special_tokens=True)
        except Exception:
            return False

        open_brackets = text.count("[")
        close_brackets = text.count("]")
        entity_count = text.count('"text"')

        if close_brackets > 0 and close_brackets >= open_brackets:
            return True
        if entity_count > self.max_entities:
            return True
        if text.strip().endswith("]") or text.strip().endswith("}]"):
            return True
        return False


def fix_entity_type(entity_type: str, threshold: float = 0.7) -> str:
    cleaned = entity_type.strip().upper()
    cleaned = unicodedata.normalize("NFKD", cleaned).encode("ascii", "ignore").decode("ascii")
    cleaned = cleaned.replace(" ", "_").replace("-", "_")
    cleaned = re.sub(r"_+", "_", cleaned)
    if cleaned in entity_types:
        return cleaned

    best_match = None
    best_ratio = 0.0
    for standard_type in entity_types:
        ratio = SequenceMatcher(None, cleaned, standard_type).ratio()
        if ratio > best_ratio and ratio >= threshold:
            best_ratio = ratio
            best_match = standard_type

    if best_match is None:
        return "UNCLEAR"
    return best_match


def fix_entity_positions(entities: List[Dict[str, Any]], input_text: str) -> List[Dict[str, Any]]:
    fixed = []
    for ent in entities:
        text = (ent.get("text") or "").strip()
        if not text:
            continue
        idx = input_text.find(text)
        if idx >= 0:
            ent = dict(ent)
            ent["start"] = idx
            ent["end"] = idx + len(text)
        fixed.append(ent)
    return fixed


def truncate_at_valid_json(text: str) -> str:
    start = text.find("[")
    end = text.rfind("]")
    if start >= 0 and end >= 0 and end > start:
        return text[start : end + 1]
    return text


def parse_json_entities(text: str) -> List[Dict[str, Any]]:
    try:
        data = json.loads(text)
    except Exception:
        return []
    if not isinstance(data, list):
        return []
    out = []
    for item in data:
        if not isinstance(item, dict):
            continue
        out.append(item)
    return out


def is_json_parsable(text: str) -> bool:
    try:
        data = json.loads(text)
    except Exception:
        return False
    return isinstance(data, list)


def validate_json_entities(response: str, input_text: str) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    cleaned = truncate_at_valid_json(response)
    raw_entities = parse_json_entities(cleaned)

    fixed = []
    for ent in raw_entities:
        fixed_ent = dict(ent)
        fixed_ent["type"] = fix_entity_type(str(ent.get("type", "")))
        fixed.append(fixed_ent)

    fixed = fix_entity_positions(fixed, input_text)
    return raw_entities, fixed


def normalize_ground_truth(ground_truth: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    normalized = []
    for ent in ground_truth:
        if not isinstance(ent, dict):
            continue
        ent_norm = dict(ent)
        ent_norm["type"] = fix_entity_type(str(ent.get("type", "")))
        normalized.append(ent_norm)
    return normalized


def compute_entity_metrics(predicted: List[Dict[str, Any]], ground_truth: List[Dict[str, Any]]) -> Dict[str, int]:
    pred_set = {(p.get("start"), p.get("end"), p.get("type")) for p in predicted}
    gold_set = {(g.get("start"), g.get("end"), g.get("type")) for g in ground_truth}

    tp_exact = len(pred_set & gold_set)
    fp_exact = len(pred_set - gold_set)
    fn_exact = len(gold_set - pred_set)

    tp_partial = 0
    matched_gold = set()
    for p in predicted:
        for i, g in enumerate(ground_truth):
            if i in matched_gold:
                continue
            if p.get("type") != g.get("type"):
                continue
            ps, pe = p.get("start"), p.get("end")
            gs, ge = g.get("start"), g.get("end")
            if ps is None or pe is None or gs is None or ge is None:
                continue
            if ps < ge and pe > gs:
                tp_partial += 1
                matched_gold.add(i)
                break

    fp_partial = max(0, len(predicted) - tp_partial)
    fn_partial = max(0, len(ground_truth) - tp_partial)

    return {
        "tp_exact": tp_exact,
        "fp_exact": fp_exact,
        "fn_exact": fn_exact,
        "tp_partial": tp_partial,
        "fp_partial": fp_partial,
        "fn_partial": fn_partial,
    }


def compute_hallucination_rate(entities: List[Dict[str, Any]], input_text: str) -> float:
    if not entities:
        return 0.0
    bad = 0
    for ent in entities:
        text = (ent.get("text") or "").strip()
        if text and text not in input_text:
            bad += 1
    return bad / len(entities)


def compute_boundary_accuracy(predicted: List[Dict[str, Any]], ground_truth: List[Dict[str, Any]]) -> float:
    if not predicted or not ground_truth:
        return 0.0
    gt_lookup = {(g.get("text"), g.get("type")): (g.get("start"), g.get("end")) for g in ground_truth}
    total = 0
    correct = 0
    for p in predicted:
        key = (p.get("text"), p.get("type"))
        if key in gt_lookup:
            total += 1
            if (p.get("start"), p.get("end")) == gt_lookup[key]:
                correct += 1
    return correct / total if total > 0 else 0.0


def compute_format_compliance(entities: List[Dict[str, Any]]) -> float:
    if not entities:
        return 0.0
    for ent in entities:
        if not all(k in ent for k in ("text", "type", "start", "end")):
            return 0.0
        if str(ent.get("type", "")).upper() not in entity_types:
            return 0.0
        if not isinstance(ent.get("start"), int) or not isinstance(ent.get("end"), int):
            return 0.0
    return 1.0


def convert_to_brat_format(entities: List[Dict[str, Any]], doc_id: str) -> str:
    lines = []
    idx = 1
    for ent in entities:
        etype = ent.get("type", "UNCLEAR")
        text = ent.get("text", "")
        start = ent.get("start", None)
        end = ent.get("end", None)
        try:
            start = int(start)
            end = int(end)
        except (TypeError, ValueError):
            continue
        if str(etype).upper() not in entity_types:
            continue
        if start < 0 or end <= start:
            continue
        lines.append(f"T{idx}\t{etype} {start} {end}\t{text}")
        idx += 1
    return "\n".join(lines)


class SafeDict(dict):
    def __missing__(self, key):
        return ""


def parse_template(path: Path) -> Dict[str, str]:
    content = path.read_text(encoding="utf-8")
    system = ""
    user = ""
    if "### SYSTEM ###" in content and "### USER ###" in content:
        parts = content.split("### SYSTEM ###", 1)[1]
        system_part, user_part = parts.split("### USER ###", 1)
        system = system_part.strip()
        user = user_part.strip()
    else:
        user = content.strip()
    return {"system": system, "user": user}


def render_template(template: str, mapping: Dict[str, Any]) -> str:
    if not template:
        return ""
    rendered = template
    for key, value in mapping.items():
        rendered = rendered.replace("{" + str(key) + "}", str(value))
    return rendered


def build_inference_formatting_fn(tokenizer, system_tmpl: str, user_tmpl: str):
    def formatting_func(example: Dict[str, Any]):
        sd = SafeDict(**example)
        sys_text = system_tmpl.format_map(sd) if system_tmpl else ""
        usr_text = user_tmpl.format_map(sd).strip()

        messages = []
        if sys_text:
            messages.append({"role": "system", "content": sys_text})
        messages.append({"role": "user", "content": usr_text})

        rendered = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        return rendered, usr_text

    return formatting_func


def parse_args():
    parser = argparse.ArgumentParser(description="Infer entities with LoRA adapters.")
    parser.add_argument("--base_model_path", type=str, required=True)
    parser.add_argument("--lora_dir", type=str, required=True)
    parser.add_argument("--output_path", type=str, default="inference_results.jsonl")
    parser.add_argument("--max_new_tokens", type=int, default=1024)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--repetition_penalty", type=float, default=1.0)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--do_sample", action="store_true")
    parser.add_argument("--no_repeat_ngram_size", type=int, default=0)
    parser.add_argument("--search_method", type=str, default="none", choices=["none", "grid", "random", "bayesian"])
    parser.add_argument("--n_trials", type=int, default=50)
    parser.add_argument("--eval_subset_size", type=int, default=50)
    parser.add_argument("--max_samples", type=int, default=0, help="Limit number of test samples (0 = all).")
    return parser.parse_args()


def main():
    args = parse_args()
    print(args.__dict__)

    base_dir = Path(__file__).resolve().parent
    template_path = base_dir / "basePromptInstruction.txt"
    test_path = str(base_dir / "instructed_prompts_test_p0.jsonl")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(args.base_model_path, use_fast=True)
    ensure_chat_template(tokenizer, args.base_model_path, os.environ.get("HF_TOKEN", ""))
    tokenizer.padding_side = "right"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    print("Loading base model...")
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    print("Model loaded.")

    print("Loading LoRA adapters...")
    model = PeftModel.from_pretrained(model, args.lora_dir, device_map="auto")
    model.eval()
    print("LoRA adapters loaded.")

    tmpl = parse_template(template_path)
    system_tmpl = tmpl["system"]
    user_tmpl = tmpl["user"]
    formatting_func = build_inference_formatting_fn(tokenizer, system_tmpl, user_tmpl)

    print("Loading test dataset...")
    dataset = load_dataset("json", data_files=test_path)["train"]
    if args.max_samples and args.max_samples > 0:
        dataset = dataset.select(range(min(args.max_samples, len(dataset))))

    if args.search_method != "none":
        print("search_method is not supported in this build; use 'none'.")
        return

    stopping_criteria = StoppingCriteriaList([
        JSONStoppingCriteria(tokenizer, max_entities=50)
    ])

    results = []
    total_counts = {"tp_exact": 0, "fp_exact": 0, "fn_exact": 0, "tp_partial": 0, "fp_partial": 0, "fn_partial": 0}

    num_json_ok_raw = 0
    num_json_ok_fixed = 0
    num_samples = 0
    halluc_rates_raw = []
    halluc_rates_fixed = []
    boundary_accs_raw = []
    boundary_accs_fixed = []
    format_ok_raw = 0
    format_total_raw = 0
    format_ok_fixed = 0
    format_total_fixed = 0

    max_time = 60

    def run_generation(max_new_tokens: int, repetition_penalty: float, no_repeat_ngram_size: int) -> Tuple[str, float]:
        start_time = time.time()
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                min_new_tokens=1,
                temperature=args.temperature,
                repetition_penalty=repetition_penalty,
                num_beams=args.num_beams,
                top_p=args.top_p if args.do_sample else None,
                do_sample=args.do_sample,
                no_repeat_ngram_size=no_repeat_ngram_size,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id,
                max_time=max_time,
                stopping_criteria=stopping_criteria,
            )
        gen_time = time.time() - start_time
        response = tokenizer.decode(outputs[0][input_len:], skip_special_tokens=True).strip()
        return response, gen_time

    print(f"Processing {len(dataset)} samples...")
    for i, example in tqdm(enumerate(dataset), total=len(dataset)):
        try:
            prompt, _ = formatting_func(example)
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            input_len = inputs["input_ids"].shape[1]

            retry_used = False
            assistant_response, generation_time = run_generation(
                args.max_new_tokens,
                args.repetition_penalty,
                args.no_repeat_ngram_size,
            )

            input_text = example.get("INPUT_TEXT", "")
            raw_entities, fixed_entities = validate_json_entities(assistant_response, input_text)
            ground_truth = normalize_ground_truth(example.get("assistant", []))

            json_ok = is_json_parsable(truncate_at_valid_json(assistant_response))
            if not json_ok:
                retry_used = True
                retry_max = min(args.max_new_tokens, 512)
                retry_rep = max(args.repetition_penalty, 1.1)
                retry_ngram = max(args.no_repeat_ngram_size, 3)
                retry_resp, retry_time = run_generation(retry_max, retry_rep, retry_ngram)
                assistant_response = retry_resp
                generation_time += retry_time
                raw_entities, fixed_entities = validate_json_entities(assistant_response, input_text)
                json_ok = is_json_parsable(truncate_at_valid_json(assistant_response))

            num_json_ok_raw += int(json_ok)
            num_json_ok_fixed += int(json_ok)
            num_samples += 1

            halluc_rate_raw = compute_hallucination_rate(raw_entities, input_text)
            halluc_rate_fixed = compute_hallucination_rate(fixed_entities, input_text)
            halluc_rates_raw.append(halluc_rate_raw)
            halluc_rates_fixed.append(halluc_rate_fixed)

            b_acc_raw = compute_boundary_accuracy(raw_entities, ground_truth)
            b_acc_fixed = compute_boundary_accuracy(fixed_entities, ground_truth)
            boundary_accs_raw.append(b_acc_raw)
            boundary_accs_fixed.append(b_acc_fixed)

            if json_ok:
                format_total_raw += 1
                format_ok_raw += int(compute_format_compliance(raw_entities) == 1.0)
                format_total_fixed += 1
                format_ok_fixed += int(compute_format_compliance(fixed_entities) == 1.0)

            metrics = compute_entity_metrics(fixed_entities, ground_truth)
            total_counts["tp_exact"] += metrics["tp_exact"]
            total_counts["fp_exact"] += metrics["fp_exact"]
            total_counts["fn_exact"] += metrics["fn_exact"]
            total_counts["tp_partial"] += metrics.get("tp_partial", 0)
            total_counts["fp_partial"] += metrics.get("fp_partial", 0)
            total_counts["fn_partial"] += metrics.get("fn_partial", 0)

            results.append(
                {
                    "sample_id": i,
                    "input": example,
                    "raw_entities": raw_entities,
                    "fixed_entities": fixed_entities,
                    "ground_truth": ground_truth,
                    "raw_response": assistant_response,
                    "generation_time": generation_time,
                    "metrics": metrics,
                    "valid_json": json_ok,
                    "retry_used": retry_used,
                }
            )

        except Exception as e:
            print(f"Error processing sample {i}: {e}")
            results.append(
                {
                    "sample_id": i,
                    "input": example,
                    "raw_entities": [],
                    "fixed_entities": [],
                    "ground_truth": normalize_ground_truth(example.get("assistant", [])),
                    "raw_response": "",
                    "generation_time": 0,
                    "metrics": {},
                    "valid_json": False,
                    "error": str(e),
                }
            )

    with open(args.output_path, "w", encoding="utf-8") as f:
        for res in results:
            f.write(json.dumps(res, ensure_ascii=False) + "\n")

    raw_json_rate = num_json_ok_raw / num_samples if num_samples > 0 else 0.0
    fixed_json_rate = num_json_ok_fixed / num_samples if num_samples > 0 else 0.0
    raw_hall_rate = sum(halluc_rates_raw) / len(halluc_rates_raw) if halluc_rates_raw else 0.0
    fixed_hall_rate = sum(halluc_rates_fixed) / len(halluc_rates_fixed) if halluc_rates_fixed else 0.0
    raw_boundary = sum(boundary_accs_raw) / len(boundary_accs_raw) if boundary_accs_raw else 0.0
    fixed_boundary = sum(boundary_accs_fixed) / len(boundary_accs_fixed) if boundary_accs_fixed else 0.0
    raw_format = format_ok_raw / format_total_raw if format_total_raw > 0 else 0.0
    fixed_format = format_ok_fixed / format_total_fixed if format_total_fixed > 0 else 0.0

    summary_metrics = {
        "json_parsing_success_rate": fixed_json_rate,
        "hallucination_rate": fixed_hall_rate,
        "boundary_accuracy": fixed_boundary,
        "format_compliance": fixed_format,
        "raw_json_parsing_success_rate": raw_json_rate,
        "raw_hallucination_rate": raw_hall_rate,
        "raw_boundary_accuracy": raw_boundary,
        "raw_format_compliance": raw_format,
        "fixed_json_parsing_success_rate": fixed_json_rate,
        "fixed_hallucination_rate": fixed_hall_rate,
        "fixed_boundary_accuracy": fixed_boundary,
        "fixed_format_compliance": fixed_format,
    }

    base_prefix = os.path.splitext(args.output_path)[0]
    sum_dir = base_prefix + "_summary.json"
    with open(sum_dir, "w", encoding="utf-8") as f:
        json.dump(summary_metrics, f, indent=2)

    brat_raw_dir = base_prefix + "_brat_raw"
    brat_fixed_dir = base_prefix + "_brat_fixed"
    brat_gold_dir = base_prefix + "_brat_gold"

    os.makedirs(brat_raw_dir, exist_ok=True)
    os.makedirs(brat_fixed_dir, exist_ok=True)
    os.makedirs(brat_gold_dir, exist_ok=True)

    for res in results:
        doc_id = f"doc_{res['sample_id']}"
        raw_brat = convert_to_brat_format(res["raw_entities"], doc_id)
        with open(os.path.join(brat_raw_dir, f"{doc_id}.ann"), "w", encoding="utf-8") as f:
            f.write(raw_brat)

        fixed_brat = convert_to_brat_format(res["fixed_entities"], doc_id)
        with open(os.path.join(brat_fixed_dir, f"{doc_id}.ann"), "w", encoding="utf-8") as f:
            f.write(fixed_brat)

        gold_brat = convert_to_brat_format(res["ground_truth"], doc_id)
        with open(os.path.join(brat_gold_dir, f"{doc_id}.ann"), "w", encoding="utf-8") as f:
            f.write(gold_brat)

    total_samples = len(results)
    def calculate_micro_metrics(tp, fp, fn):
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        return precision, recall, f1

    prec_exact, rec_exact, f1_exact = calculate_micro_metrics(
        total_counts["tp_exact"], total_counts["fp_exact"], total_counts["fn_exact"]
    )
    prec_partial, rec_partial, f1_partial = calculate_micro_metrics(
        total_counts["tp_partial"], total_counts["fp_partial"], total_counts["fn_partial"]
    )

    print("\n" + "=" * 60)
    print("ENHANCED INFERENCE RESULTS")
    print("=" * 60)
    print(f"Total samples: {total_samples}")
    print(f"Results saved to: {args.output_path}")
    print(f"Pred BRAT raw dir: {brat_raw_dir}")
    print(f"Pred BRAT fixed dir: {brat_fixed_dir}")
    print(f"Gold BRAT dir: {brat_gold_dir}")
    print("\nOVERALL METRICS:")
    print(f"  Exact Match F1:     {f1_exact:.3f}")
    print(f"  Exact Match Precision: {prec_exact:.3f}")
    print(f"  Exact Match Recall:    {rec_exact:.3f}")
    print(f"  Partial Match F1:      {f1_partial:.3f}")
    print(f"  Partial Match Precision: {prec_partial:.3f}")
    print(f"  Partial Match Recall:    {rec_partial:.3f}")
    print("\nAUX METRICS (RAW):")
    print(f"  JSON parse success: {raw_json_rate:.3f}")
    print(f"  Hallucination rate: {raw_hall_rate:.3f}")
    print(f"  Boundary accuracy:  {raw_boundary:.3f}")
    print(f"  Format compliance:  {raw_format:.3f}")
    print("\nAUX METRICS (FIXED):")
    print(f"  JSON parse success: {fixed_json_rate:.3f}")
    print(f"  Hallucination rate: {fixed_hall_rate:.3f}")
    print(f"  Boundary accuracy:  {fixed_boundary:.3f}")
    print(f"  Format compliance:  {fixed_format:.3f}")


if __name__ == "__main__":
    main()
