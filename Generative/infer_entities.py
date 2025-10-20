#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import time
import json
import torch
import optuna
from tqdm import tqdm
from peft import PeftModel
from datasets import load_dataset
from difflib import SequenceMatcher
from typing import Dict, List, Tuple
from sklearn.model_selection import ParameterGrid
from transformers import AutoTokenizer, AutoModelForCausalLM, StoppingCriteria

entity_types = ["PROTEINAS", "NORMALIZABLES", "NO_NORMALIZABLES", "UNCLEAR"]

class JSONStoppingCriteria(StoppingCriteria):
    """Custom stopping criteria to detect JSON array completion."""
    
    def __init__(self, tokenizer, max_entities=50):
        self.tokenizer = tokenizer
        self.max_entities = max_entities
        
    def __call__(self, input_ids, scores, **kwargs):
        if input_ids.shape[1] < 10:
            return False
            
        last_tokens = input_ids[0][-30:]
        try:
            text = self.tokenizer.decode(last_tokens, skip_special_tokens=True)
        except:
            return False
        
        open_brackets = text.count('[')
        close_brackets = text.count(']')
        entity_count = text.count('"text"')
        
        if close_brackets > 0 and close_brackets >= open_brackets:
            return True
        if entity_count > self.max_entities:
            return True
        if (text.strip().endswith('"]') or 
            text.strip().endswith('}]') or
            text.strip().endswith(']')):
            return True
        if text.count('TYPE') > 5:
            return True
            
        return False

def fix_entity_type(entity_type: str, threshold: float = 0.7) -> str:
    """
    Fix entity type using string similarity matching.
    """
    
    # Clean the input
    cleaned = entity_type.strip().upper()
    
    # If it's already perfect, return it
    if cleaned in entity_types:
        return cleaned
    
    # Find the most similar standard type
    best_match = None
    best_ratio = 0
    
    for standard_type in entity_types:
        ratio = SequenceMatcher(None, cleaned, standard_type).ratio()
        if ratio > best_ratio and ratio >= threshold:
            best_ratio = ratio
            best_match = standard_type

    print(f"Original entity type: '{entity_type}'")
    print(f"Fixed entity type: '{best_match if best_match else 'UNCLEAR'}' (best ratio: {best_ratio:.2f})")
    
    # Return best match or default to UNCLEAR
    return best_match if best_match else "UNCLEAR"

def fix_entity_positions(entities: List[Dict], input_text: str) -> List[Dict]:
    """
    Fix incorrect entity positions by finding actual text positions.
    This addresses the position calculation errors from the model.
    """
    fixed_entities = []
    
    for entity in entities:
        text = entity.get("text", "").strip()
        entity_type = entity.get("type", "").strip()
        
        if not text:
            continue
            
        # Find all occurrences of this text in the input
        positions = []
        start = 0
        while True:
            pos = input_text.find(text, start)
            if pos == -1:
                break
            positions.append((pos, pos + len(text)))
            start = pos + 1
        
        # If we found the text, use the first occurrence
        # (In practice, you might want more sophisticated matching)
        if positions:
            start, end = positions[0]
            if entity_type not in entity_types:
                entity_type = fix_entity_type(entity_type)
            fixed_entities.append({
                "text": text,
                "type": entity_type,
                "start": start,
                "end": end
            })
        else:
            # Text not found - might be a hallucination or OCR error
            # Still include but mark positions as invalid
            if entity_type not in entity_types:
                entity_type = fix_entity_type(entity_type)

            fixed_entities.append({
                "text": text,
                "type": fix_entity_type(entity_type),
                "start": -1,
                "end": -1
            })
    
    return fixed_entities

def validate_json_entities(response: str, input_text: str) -> Tuple[List[Dict], List[Dict]]:
    """
    Enhanced validation with position fixing.
    Returns: (raw_entities, fixed_entities)
    """
    try:
        # First, try to truncate at valid JSON boundary
        truncated = truncate_at_valid_json(response.strip())
        parsed = json.loads(truncated)
        
        if not isinstance(parsed, list):
            return [], []
        
        # Extract raw entities (as predicted by model)
        raw_entities = []
        for item in parsed:
            if not isinstance(item, dict):
                continue
                
            # Extract fields with type conversion
            try:
                text = str(item.get("text", "")).strip()
                entity_type = str(item.get("type", "")).strip()
                
                # Handle both string and int positions
                start = item.get("start")
                end = item.get("end")
                
                # Convert string positions to int if possible
                if isinstance(start, str):
                    try:
                        start = int(start.strip())
                    except ValueError:
                        start = -1
                        
                if isinstance(end, str):
                    try:
                        end = int(end.strip())
                    except ValueError:
                        end = -1
                
                if text and entity_type:
                    raw_entities.append({
                        "text": text,
                        "type": entity_type,
                        "start": start if isinstance(start, int) else -1,
                        "end": end if isinstance(end, int) else -1
                    })
                    
            except (ValueError, TypeError):
                continue
        
        # Fix positions and types
        fixed_entities = fix_entity_positions(raw_entities, input_text)
        
        return raw_entities, fixed_entities
        
    except (json.JSONDecodeError, ValueError, TypeError) as e:
        print(f"JSON parsing error: {e}")
        return [], []

def truncate_at_valid_json(text: str) -> str:
    """Truncate text at the first complete, valid JSON array."""
    try:
        bracket_count = 0
        in_string = False
        escape_next = False
        
        for i, char in enumerate(text):
            if escape_next:
                escape_next = False
                continue
            
            if char == '\\':
                escape_next = True
                continue
                
            if char == '"' and not escape_next:
                in_string = not in_string
                continue
                
            if not in_string:
                if char == '[':
                    bracket_count += 1
                elif char == ']':
                    bracket_count -= 1
                    if bracket_count == 0:
                        potential_json = text[:i+1]
                        try:
                            json.loads(potential_json)
                            return potential_json
                        except json.JSONDecodeError:
                            continue
                        
    except Exception:
        pass
    
    return text

def compute_entity_metrics(predicted: List[Dict], ground_truth: List[Dict]) -> Dict[str, float]:
    """Compute precision, recall, F1 for entity extraction."""
    
    # Convert to sets for comparison (text + type + position)
    def to_tuple(entity):
        return (entity["text"], entity["type"], entity["start"], entity["end"])
    
    pred_set = {to_tuple(e) for e in predicted if e["start"] != -1}
    gt_set = {to_tuple(e) for e in ground_truth}
    
    # Exact match
    tp_exact = len(pred_set.intersection(gt_set))
    fp_exact = len(pred_set - gt_set)
    fn_exact = len(gt_set - pred_set)
    
    precision_exact = tp_exact / len(pred_set) if pred_set else 0
    recall_exact = tp_exact / len(gt_set) if gt_set else 0
    f1_exact = 2 * precision_exact * recall_exact / (precision_exact + recall_exact) if (precision_exact + recall_exact) > 0 else 0
    
    # Text + type match (ignore position)
    pred_text_type = {(e["text"], e["type"]) for e in predicted if e["start"] != -1}
    gt_text_type = {(e["text"], e["type"]) for e in ground_truth}

    tp_partial = len(pred_text_type.intersection(gt_text_type))
    fp_partial = len(pred_text_type - gt_text_type)
    fn_partial = len(gt_text_type - pred_text_type)
    
    precision_partial = tp_partial / len(pred_text_type) if pred_text_type else 0
    recall_partial = tp_partial / len(gt_text_type) if gt_text_type else 0
    f1_partial = 2 * precision_partial * recall_partial / (precision_partial + recall_partial) if (precision_partial + recall_partial) > 0 else 0
    
    return {
        "precision_exact": precision_exact,
        "recall_exact": recall_exact, 
        "f1_exact": f1_exact,
        "precision_partial": precision_partial,
        "recall_partial": recall_partial,
        "f1_partial": f1_partial,
        "tp_exact": tp_exact,
        "fp_exact": fp_exact,
        "fn_exact": fn_exact,
        "tp_partial": tp_partial,
        "fp_partial": fp_partial,
        "fn_partial": fn_partial
    }

def parse_template(path: str) -> Dict[str, str]:
    with open(path, "r", encoding="utf-8") as f:
        content = f.read()

    sys_match = re.search(r"###\s*SYSTEM\s*###\s*(.*?)(?=###\s*USER\s*###|$)", content, flags=re.S | re.I)
    usr_match = re.search(r"###\s*USER\s*###\s*(.*)$", content, flags=re.S | re.I)

    system_tmpl = sys_match.group(1).strip() if sys_match else ""
    user_tmpl = usr_match.group(1).strip() if usr_match else ""

    if not user_tmpl:
        raise ValueError("Template must include a '### USER ###' section with content.")

    return {"system": system_tmpl, "user": user_tmpl}

class SafeDict(dict):
    def __missing__(self, key):
        return "{" + key + "}"

def build_inference_formatting_fn(tokenizer, system_tmpl, user_tmpl):
    def formatting_func(example):
        sd = SafeDict(**example)
        sys_text = system_tmpl.format_map(sd) if system_tmpl else ""
        usr_text = user_tmpl.format_map(sd).strip()

        messages = []
        if sys_text:
            messages.append({"role": "system", "content": sys_text})
        messages.append({"role": "user", "content": usr_text})

        rendered = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        return rendered, usr_text

    return formatting_func

class HyperparameterOptimizer:
    def __init__(self, model, tokenizer, formatting_func, test_dataset, device):
        self.model = model
        self.tokenizer = tokenizer
        self.formatting_func = formatting_func
        self.test_dataset = test_dataset
        self.device = device
        self.best_score = 0
        self.best_params = {}

    def evaluate_configuration(self, params: Dict, subset_size: int = 50) -> Dict[str, float]:
        """Evaluate a single hyperparameter configuration with micro-averaging."""
        eval_dataset = self.test_dataset.select(range(min(subset_size, len(self.test_dataset))))
        
        total_counts = {
            "tp_exact": 0, "fp_exact": 0, "fn_exact": 0,
            "tp_partial": 0, "fp_partial": 0, "fn_partial": 0
        }
        
        successful_samples = 0
        
        for i, example in enumerate(eval_dataset):
            try:
                prompt, _ = self.formatting_func(example)
                inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
                input_len = inputs["input_ids"].shape[1]
                
                # Generate with timeout protection
                with torch.no_grad():
                    outputs = self.model.generate(
                        **inputs,
                        max_new_tokens=min(params.get('max_new_tokens', 1342), 1500),  # Cap tokens
                        temperature=params.get('temperature', 0.1),
                        repetition_penalty=params.get('repetition_penalty', 1.3),
                        num_beams=params.get('num_beams', 4),
                        top_p=params.get('top_p', 0.9) if params.get('do_sample', False) else None,
                        do_sample=params.get('do_sample', False),
                        no_repeat_ngram_size=params.get('no_repeat_ngram_size', 4),
                        early_stopping=True,
                        eos_token_id=self.tokenizer.eos_token_id,
                        pad_token_id=self.tokenizer.pad_token_id,
                        max_time=30  # 30 second timeout per generation
                    )

                assistant_response = self.tokenizer.decode(
                    outputs[0][input_len:], skip_special_tokens=True
                ).strip()

                input_text = example["INPUT_TEXT"]
                _, fixed_entities = validate_json_entities(assistant_response, input_text)
                ground_truth = example.get("assistant", [])
                metrics = compute_entity_metrics(fixed_entities, ground_truth)

                # Accumulate raw counts for micro-averaging
                for key in ["tp_exact", "fp_exact", "fn_exact", "tp_partial", "fp_partial", "fn_partial"]:
                    total_counts[key] += metrics.get(key, 0)
                
                successful_samples += 1

            except Exception as e:
                print(f"  Sample {i} failed: {str(e)[:100]}", flush=True)
                continue
        
        # Calculate micro-averaged metrics
        def calculate_micro_metrics(tp, fp, fn):
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            return precision, recall, f1

        prec_exact, rec_exact, f1_exact = calculate_micro_metrics(
            total_counts["tp_exact"], total_counts["fp_exact"], total_counts["fn_exact"])
        prec_partial, rec_partial, f1_partial = calculate_micro_metrics(
            total_counts["tp_partial"], total_counts["fp_partial"], total_counts["fn_partial"])

        print(f"  Processed {successful_samples}/{len(eval_dataset)} samples", flush=True)
        
        return {
            "precision_exact": prec_exact, "recall_exact": rec_exact, "f1_exact": f1_exact,
            "precision_partial": prec_partial, "recall_partial": rec_partial, "f1_partial": f1_partial
        }

    
    def grid_search(self, param_grid: Dict, subset_size: int = 50):
        """Perform grid search over hyperparameters."""
        
        print("Starting Grid Search...")
        grid = ParameterGrid(param_grid)
        
        results = []
        for i, params in enumerate(grid):
            print(f"Evaluating configuration {i+1}/{len(grid)}: {params}")
            
            metrics = self.evaluate_configuration(params, subset_size)
            score = metrics['f1_exact']  # Use exact F1 as primary metric
            
            results.append({
                'params': params,
                'metrics': metrics,
                'score': score
            })
            
            if score > self.best_score:
                self.best_score = score
                self.best_params = params.copy()
                print(f"New best score: {score:.4f} with params: {params}")
        
        return results

    def random_search(self, param_space: Dict, n_trials: int = 50, subset_size: int = 50):
        """Perform random search over hyperparameters."""
        
        import random
        
        print(f"Starting Random Search with {n_trials} trials...")
        results = []
        
        for trial in range(n_trials):
            # Sample random parameters
            params = {}
            for key, values in param_space.items():
                params[key] = random.choice(values)
            
            print(f"Trial {trial+1}/{n_trials}: {params}")
            
            metrics = self.evaluate_configuration(params, subset_size)
            score = metrics['f1_exact']
            
            results.append({
                'params': params,
                'metrics': metrics,
                'score': score
            })
            
            if score > self.best_score:
                self.best_score = score
                self.best_params = params.copy()
                print(f"New best score: {score:.4f}")
        
        return results

    def bayesian_optimization(self, n_trials: int = 100, subset_size: int = 50):
        """Use Optuna for Bayesian optimization with robust error handling."""
        import time
        import traceback
        
        def objective(trial):
            try:
                params = {
                    'temperature': trial.suggest_float('temperature', 0.0, 0.5),
                    'repetition_penalty': trial.suggest_float('repetition_penalty', 1.0, 2.0),
                    'num_beams': trial.suggest_int('num_beams', 1, 8),
                    'top_p': trial.suggest_float('top_p', 0.7, 1.0),
                    'do_sample': trial.suggest_categorical('do_sample', [True, False]),
                    'no_repeat_ngram_size': trial.suggest_int('no_repeat_ngram_size', 2, 6),
                    'max_new_tokens': trial.suggest_int('max_new_tokens', 800, 1500)  # Reduced max
                }
                
                print(f"[TRIAL {trial.number:2d}] Starting with params: {params}", flush=True)
                start_time = time.time()
                
                metrics = self.evaluate_configuration(params, subset_size)
                
                duration = time.time() - start_time
                f1_score = metrics.get('f1_exact', 0.0)
                
                print(f"[TRIAL {trial.number:2d}] Completed in {duration:.1f}s, F1_exact: {f1_score:.4f}", flush=True)
                
                return f1_score

            except Exception as e:
                print(f"[TRIAL {trial.number:2d}] FAILED: {str(e)}", flush=True)
                traceback.print_exc()
                return 0.0

        def progress_callback(study, trial):
            print(f"[CALLBACK] Trial {trial.number} finished. Current best: {study.best_value:.4f}", flush=True)

        print("Starting Bayesian optimization with Optuna...", flush=True)
        
        try:
            study = optuna.create_study(direction='maximize')
            study.optimize(objective, n_trials=n_trials, callbacks=[progress_callback])
            
            print("✓ Optimization completed successfully!", flush=True)
            
            self.best_params = study.best_params.copy()
            self.best_score = study.best_value
            
            print(f"✓ Best trial: {study.best_trial.number}", flush=True)
            print(f"✓ Best F1_exact: {self.best_score:.4f}", flush=True)
            print(f"✓ Best parameters: {self.best_params}", flush=True)
            
            return study
            
        except Exception as e:
            print(f"ERROR in Bayesian optimization: {e}", flush=True)
            traceback.print_exc()
            self.best_params = {}
            self.best_score = 0.0
            return None

def convert_to_brat_format(entities, doc_id):
    """Convert entities to BRAT annotation format for official evaluation."""
    brat_lines = []
    for i, entity in enumerate(entities):
        if entity["start"] != -1:  # Only valid positions
            line = f"T{i+1}\t{entity['type']} {entity['start']} {entity['end']}\t{entity['text']}"
            brat_lines.append(line)
    return '\n'.join(brat_lines)

def is_json_parsable(text):
    try:
        json.loads(text)
        return True
    except Exception:
        return False

def check_format_compliance(entity):
    # All required fields
    keys_ok = set(entity.keys()) == {"text", "type", "start", "end"}
    types_ok = (
        isinstance(entity.get("text"), str)
        and entity.get("type") in entity_types
        and isinstance(entity.get("start"), int)
        and isinstance(entity.get("end"), int)
    )
    return keys_ok and types_ok

def compute_hallucination_rate(predicted, input_text):
    if len(predicted) == 0:
        return 0.0
    hallucinated = 0
    for e in predicted:
        if e["start"] == -1:
            hallucinated += 1
        elif e["text"] not in input_text:
            hallucinated += 1
    return hallucinated / len(predicted)

def compute_boundary_accuracy(predicted, ground_truth):
    if not ground_truth or not predicted:
        return 0.0
    # Exact boundaries among matching predictions
    total_matches = 0
    boundary_matches = 0
    gt_tuples = {(gt["text"], gt["type"], gt["start"], gt["end"]) for gt in ground_truth}
    gt_lookup = {(gt["text"], gt["type"]): (gt["start"], gt["end"]) for gt in ground_truth}
    for pred in predicted:
        key = (pred["text"], pred["type"])
        if key in gt_lookup:
            total_matches += 1
            if (pred["start"], pred["end"]) == gt_lookup[key]:
                boundary_matches += 1
    return boundary_matches / total_matches if total_matches > 0 else 0.0

def compute_format_compliance(entities):
    if not entities:
        return 0.0
    return float(all(check_format_compliance(e) for e in entities))

def main():

    #Read the argument --base_model_path, --lora_dir, --max_new_tokens
    import argparse
    import json

    parser = argparse.ArgumentParser(description="Entity Inference")
    parser.add_argument("--base_model_path", type=str, default="llama-3.2-3b-instruct", help="Path to the base model")
    parser.add_argument("--lora_dir", type=str, default="outputs/llama-3.2-3b-instruct-lora", help="Path to the LoRA directory")
    parser.add_argument("--output_path", type=str, default="inference_results.jsonl", help="Path to save inference results")

    # Generation arguments
    parser.add_argument("--max_new_tokens", type=int, default=1342, help="Maximum number of new tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.1, help="Temperature for generation")
    parser.add_argument("--repetition_penalty", type=float, default=1.3, help="Repetition penalty")
    parser.add_argument("--num_beams", type=int, default=4, help="Number of beams for beam search")
    parser.add_argument("--top_p", type=float, default=0.9, help="Top-p sampling parameter")
    parser.add_argument("--do_sample", action='store_true', help="Whether to use sampling")
    parser.add_argument("--no_repeat_ngram_size", type=int, default=4, help="No repeat ngram size")

    
    # Hyperparameter search arguments
    parser.add_argument("--search_method", type=str, default="none", choices=["none", "grid", "random", "bayesian"], help="Hyperparameter search method")
    parser.add_argument("--n_trials", type=int, default=50, help="Number of trials for random/bayesian search")
    parser.add_argument("--eval_subset_size", type=int, default=50, help="Subset size for evaluation during search")

    args = parser.parse_args()

    # Print the arguments
    print(args.__dict__)

    # Paths
    template_path = "Instruction/basePromptInstruction.txt"
    test_path = "Instruction/instructed_prompts_test_p0.jsonl"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(args.base_model_path, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    print("Loading base model...")
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model_path,
        dtype=torch.bfloat16,
        device_map="auto",
    )
    print("Model loaded.")

    print("Loading LoRA adapters...")
    model = PeftModel.from_pretrained(
        model,
        args.lora_dir,
        device_map="auto"
    )
    model.eval()
    print("LoRA adapters loaded.")

    # Parse template and create formatting function
    tmpl = parse_template(template_path)
    system_tmpl = tmpl["system"]
    user_tmpl = tmpl["user"]
    formatting_func = build_inference_formatting_fn(tokenizer, system_tmpl, user_tmpl)

    # Load test dataset
    print("Loading test dataset...")
    dataset = load_dataset("json", data_files=test_path)["train"]


    if args.search_method != "none":
        # Create optimizer
        optimizer = HyperparameterOptimizer(
            model=model,
            tokenizer=tokenizer,
            formatting_func=formatting_func,
            test_dataset=dataset,
            device=device
        )

        # Define search space
        if args.search_method == "grid":
            # Smaller grid for feasible search
            param_grid = {
                'temperature': [0.0, 0.1, 0.2],
                'repetition_penalty': [1.1, 1.3, 1.5],
                'num_beams': [2, 4, 6],
                'do_sample': [False, True]
            }
            results = optimizer.grid_search(param_grid, args.eval_subset_size)
            
        elif args.search_method == "random":
            param_space = {
                'temperature': [0.0, 0.05, 0.1, 0.15, 0.2, 0.3],
                'repetition_penalty': [1.1, 1.3, 1.5],
                'num_beams': [1, 4, 6],
                'top_p': [0.85, 0.9, 0.95, 1.0],
                'do_sample': [True, False],
                'no_repeat_ngram_size': [3, 4, 5],
                'max_new_tokens': [981, 1342, 1793]
            }
            results = optimizer.random_search(param_space, args.n_trials, args.eval_subset_size)
            
        elif args.search_method == "bayesian":
            study = optimizer.bayesian_optimization(args.n_trials, args.eval_subset_size)
        
        # Print best configuration
        print(f"\n{'='*60}")
        print(f"HYPERPARAMETER SEARCH RESULTS")
        print(f"{'='*60}")
        print(f"Best score (F1 Exact): {optimizer.best_score:.4f}")
        print(f"Best parameters:")
        for key, value in optimizer.best_params.items():
            print(f"  {key}: {value}")
        
        # Final evaluation with best parameters on full dataset
        print(f"\n{'='*60}")
        print("FINAL EVALUATION WITH BEST PARAMETERS")
        print(f"{'='*60}")
        
        final_metrics = optimizer.evaluate_configuration(
            optimizer.best_params, 
            subset_size=len(dataset)
        )
        
        print("Final metrics on full dataset:")
        for key, value in final_metrics.items():
            print(f"  {key}: {value:.4f}")
        
        # Save best parameters
        with open(args.output_path, "w") as f:
            json.dump({
                'best_params': optimizer.best_params,
                'best_score': optimizer.best_score,
                'final_metrics': final_metrics
            }, f, indent=2)

    else:
    
        # Optimized parameters for entity extraction accuracy
        max_entities       = 30
        max_time           = 60       
        
        # Process full dataset or subset
        #dataset = dataset.select(range(10))  # Test with 10 samples first
        print(f"Processing {len(dataset)} samples...")

        # Inference loop with enhanced validation
        results = []
        total_counts = {
            "tp_exact": 0, "fp_exact": 0, "fn_exact": 0,
            "tp_partial": 0, "fp_partial": 0, "fn_partial": 0
        }

        num_json_ok = 0
        num_samples = 0
        halluc_rates = []
        boundary_accs = []
        format_ok = 0
        format_total = 0
        
        for i, example in tqdm(enumerate(dataset), total=len(dataset)):
            
            try:
                # Format prompt
                prompt, _ = formatting_func(example)
                inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
                input_len = inputs["input_ids"].shape[1]

                # Generate with optimized parameters
                start_time = time.time()
                with torch.no_grad():
                    outputs = model.generate(
                        **inputs,
                        max_new_tokens=args.max_new_tokens,
                        temperature=args.temperature,
                        repetition_penalty=args.repetition_penalty,
                        num_beams=args.num_beams,
                        top_p=args.top_p if args.do_sample else None,
                        do_sample=args.do_sample,
                        no_repeat_ngram_size=args.no_repeat_ngram_size,
                        early_stopping=True,
                        eos_token_id=tokenizer.eos_token_id,
                        pad_token_id=tokenizer.pad_token_id,
                        max_time=max_time
                    )
                
                generation_time = time.time() - start_time

                # Decode response
                assistant_response = tokenizer.decode(
                    outputs[0][input_len:], 
                    skip_special_tokens=True
                ).strip()

                # Enhanced validation with position fixing
                input_text = example["INPUT_TEXT"]
                raw_entities, fixed_entities = validate_json_entities(assistant_response, input_text)
                
                # Get ground truth
                ground_truth = example.get("assistant", [])

                # Compute additional metrics
                json_ok = is_json_parsable(truncate_at_valid_json(assistant_response))
                num_json_ok += int(json_ok)
                num_samples += 1

                halluc_rate = compute_hallucination_rate(fixed_entities, input_text)
                halluc_rates.append(halluc_rate)

                b_acc = compute_boundary_accuracy(fixed_entities, ground_truth)
                boundary_accs.append(b_acc)

                if json_ok:
                    format_total += 1
                    format_ok += int(compute_format_compliance(fixed_entities) == 1.0)
                
                # Compute metrics
                metrics = compute_entity_metrics(fixed_entities, ground_truth)

                # Accumulate raw counts for micro-averaging
                total_counts["tp_exact"] += metrics["tp_exact"]
                total_counts["fp_exact"] += metrics["fp_exact"] 
                total_counts["fn_exact"] += metrics["fn_exact"]
                total_counts["tp_partial"] += metrics.get("tp_partial", 0)
                total_counts["fp_partial"] += metrics.get("fp_partial", 0)
                total_counts["fn_partial"] += metrics.get("fn_partial", 0)

                # Store detailed results
                results.append({
                    "sample_id": i,
                    "input": example,
                    "raw_entities": raw_entities,
                    "fixed_entities": fixed_entities,
                    "ground_truth": ground_truth,
                    "raw_response": assistant_response,
                    "generation_time": generation_time,
                    "metrics": metrics,
                    "valid_json": len(raw_entities) > 0 or assistant_response.strip() == "[]"
                })

            except Exception as e:
                print(f"Error processing sample {i}: {e}")
                results.append({
                    "sample_id": i,
                    "input": example,
                    "raw_entities": [],
                    "fixed_entities": [],
                    "ground_truth": example.get("assistant", []),
                    "raw_response": "",
                    "generation_time": 0,
                    "metrics": {},
                    "valid_json": False,
                    "error": str(e)
                })

        # Save results
        with open(args.output_path, "w", encoding="utf-8") as f:
            for res in results:
                f.write(json.dumps(res, ensure_ascii=False) + "\n")

        # Save summary metrics
        summary_metrics = {
            "json_parsing_success_rate": num_json_ok / num_samples if num_samples > 0 else 0,
            "hallucination_rate": sum(halluc_rates) / len(halluc_rates) if halluc_rates else 0,
            "boundary_accuracy": sum(boundary_accs) / len(boundary_accs) if boundary_accs else 0,
            "format_compliance": format_ok / format_total if format_total > 0 else 0,
        }

        sum_dir = args.output_path.split('.')[0] + "_summary.json"

        with open(sum_dir, "w") as f:
            json.dump(summary_metrics, f, indent=2)

        # Export BRAT annotations for official evaluation
        brat_dir = args.output_path.split('.')[0] + "_brat"

        if not os.path.exists(brat_dir):
            os.makedirs(brat_dir)

        for res in results:
            doc_id = f"doc_{res['sample_id']}"
            brat_content = convert_to_brat_format(res["fixed_entities"], doc_id)
            with open(os.path.join(brat_dir, f"{doc_id}.ann"), "w", encoding="utf-8") as f:
                f.write(brat_content)
        # Print comprehensive summary
        total_samples = len(results)
        
        # Calculate micro-averaged metrics
        def calculate_micro_metrics(tp, fp, fn):
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            return precision, recall, f1

        prec_exact, rec_exact, f1_exact = calculate_micro_metrics(
            total_counts["tp_exact"], total_counts["fp_exact"], total_counts["fn_exact"])
        prec_partial, rec_partial, f1_partial = calculate_micro_metrics(
            total_counts["tp_partial"], total_counts["fp_partial"], total_counts["fn_partial"])

        avg_metrics = {
            "precision_exact": prec_exact, "recall_exact": rec_exact, "f1_exact": f1_exact,
            "precision_partial": prec_partial, "recall_partial": rec_partial, "f1_partial": f1_partial
        }
        
        print(f"\n{'='*60}")
        print(f"ENHANCED INFERENCE RESULTS")
        print(f"{'='*60}")
        print(f"Total samples: {total_samples}")
        print(f"Results saved to: {args.output_path}")
        print(f"\nOVERALL METRICS:")
        print(f"  Exact Match F1:     {avg_metrics['f1_exact']:.3f}")
        print(f"  Exact Match Precision: {avg_metrics['precision_exact']:.3f}")
        print(f"  Exact Match Recall:    {avg_metrics['recall_exact']:.3f}")
        print(f"  Partial Match F1:      {avg_metrics['f1_partial']:.3f}")
        print(f"  Partial Match Precision: {avg_metrics['precision_partial']:.3f}")
        print(f"  Partial Match Recall:    {avg_metrics['recall_partial']:.3f}")


if __name__ == "__main__":
    main()
