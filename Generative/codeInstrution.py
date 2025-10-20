#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import json
import torch
import argparse
import numpy as np
import time
from datasets import load_dataset
from trl import SFTTrainer, SFTConfig
from peft import LoraConfig, get_peft_model
from typing import Dict, List, Tuple
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    EarlyStoppingCallback,
    StoppingCriteria,
    StoppingCriteriaList
)


def is_rank0():
    try:
        return not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0
    except Exception:
        return True


class JSONStoppingCriteria(StoppingCriteria):
    """Custom stopping criteria for JSON completion during validation."""
    
    def __init__(self, tokenizer, max_entities=30):
        self.tokenizer = tokenizer
        self.max_entities = max_entities
        
    def __call__(self, input_ids, scores, **kwargs):
        if input_ids.shape[1] < 10:
            return False
            
        last_tokens = input_ids[0][-30:]
        try:
            text = self.processing_class.decode(last_tokens, skip_special_tokens=True)
        except:
            return False
        
        open_brackets = text.count('[')
        close_brackets = text.count(']')
        entity_count = text.count('"text"')
        
        # Stop conditions for JSON completion
        if close_brackets > 0 and close_brackets >= open_brackets:
            return True
        if entity_count > self.max_entities:
            return True
        if text.strip().endswith(']'):
            return True
        if text.count('TYPE') > 5:  # Detect degradation
            return True
            
        return False


def fix_entity_type(entity_type: str) -> str:
    """Fix common entity type errors and normalize to standard types."""
    type_corrections = {
        "PROTEÌNAS": "PROTEINAS", "PROTÈINAS": "PROTEINAS", "PROTÊINAS": "PROTEINAS", 
        "PROTIÈNAS": "PROTEINAS", "PROTE ÎNAS": "PROTEINAS", "PROT ÈINAS": "PROTEINAS",
        "PROTINAS": "PROTEINAS", "PROTEINAS ": "PROTEINAS", " PROTEINAS": "PROTEINAS",
        
        "NORMALIZABLENS": "NORMALIZABLES", "NORMALÍZABLES": "NORMALIZABLES",
        "NORMALÍZAABLES": "NORMALIZABLES", "NOMINALIZED": "NORMALIZABLES",
        "NORMALISABLES": "NORMALIZABLES", "NO NORMALIZABLES": "NO_NORMALIZABLES",
        "NO NORMALÍZARES": "NO_NORMALIZABLES", "NO NORMALISABLES": "NO_NORMALIZABLES",
        
        "NCLEA": "UNCLEAR", "NCLER": "UNCLEAR", "NCLEAR": "UNCLEAR", "UCLER": "UNCLEAR",
        "UCLEA": "UNCLEAR", "UNC Leer": "UNCLEAR", "UNCLER": "UNCLEAR", "UNCLEA": "UNCLEAR",
        "UNCLEAR ": "UNCLEAR", " UNCLEAR": "UNCLEAR"
    }
    
    cleaned = entity_type.strip().upper()
    return type_corrections.get(cleaned, cleaned if cleaned in ["PROTEINAS", "NORMALIZABLES", "NO_NORMALIZABLES", "UNCLEAR"] else "UNCLEAR")


def fix_entity_positions(entities: List[Dict], input_text: str) -> List[Dict]:
    """Fix incorrect entity positions by finding actual text positions."""
    fixed_entities = []
    
    for entity in entities:
        text = entity.get("text", "").strip()
        entity_type = entity.get("type", "").strip()
        
        if not text:
            continue
            
        # Find first occurrence of text
        pos = input_text.find(text)
        if pos != -1:
            fixed_entities.append({
                "text": text,
                "type": fix_entity_type(entity_type),
                "start": pos,
                "end": pos + len(text)
            })
    
    return fixed_entities


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


def parse_entities_from_response(response: str, input_text: str) -> List[Dict]:
    """Parse and fix entities from model response."""
    try:
        truncated = truncate_at_valid_json(response.strip())
        parsed = json.loads(truncated)
        
        if not isinstance(parsed, list):
            return []
        
        # Extract raw entities
        raw_entities = []
        for item in parsed:
            if not isinstance(item, dict):
                continue
                
            text = str(item.get("text", "")).strip()
            entity_type = str(item.get("type", "")).strip()
            
            # Handle string positions
            start = item.get("start")
            end = item.get("end")
            
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
        
        # Fix positions and types
        return fix_entity_positions(raw_entities, input_text)
        
    except (json.JSONDecodeError, ValueError, TypeError):
        return []


def compute_entity_metrics(predicted: List[Dict], ground_truth: List[Dict]) -> Dict[str, float]:
    """Compute entity extraction metrics."""
    def to_tuple(entity):
        return (entity["text"], entity["type"], entity["start"], entity["end"])
    
    pred_set = {to_tuple(e) for e in predicted}
    gt_set = {to_tuple(e) for e in ground_truth}
    
    # Exact match metrics
    tp_exact = len(pred_set.intersection(gt_set))
    fp_exact = len(pred_set - gt_set)
    fn_exact = len(gt_set - pred_set)
    
    precision_exact = tp_exact / len(pred_set) if pred_set else 0
    recall_exact = tp_exact / len(gt_set) if gt_set else 0
    f1_exact = 2 * precision_exact * recall_exact / (precision_exact + recall_exact) if (precision_exact + recall_exact) > 0 else 0
    
    # Partial match (text + type, ignore position)
    pred_text_type = {(e["text"], e["type"]) for e in predicted}
    gt_text_type = {(e["text"], e["type"]) for e in ground_truth}
    
    tp_partial = len(pred_text_type.intersection(gt_text_type))
    fp_partial = len(pred_text_type - gt_text_type)
    fn_partial = len(gt_text_type - pred_text_type)
    
    precision_partial = tp_partial / len(pred_text_type) if pred_text_type else 0
    recall_partial = tp_partial / len(gt_text_type) if gt_text_type else 0
    f1_partial = 2 * precision_partial * recall_partial / (precision_partial + recall_partial) if (precision_partial + recall_partial) > 0 else 0
    
    return {
        "entity_f1_exact": f1_exact,
        "entity_precision_exact": precision_exact,
        "entity_recall_exact": recall_exact,
        "entity_f1_partial": f1_partial,
        "entity_precision_partial": precision_partial,
        "entity_recall_partial": recall_partial,
        "entity_count_pred": len(predicted),
        "entity_count_true": len(ground_truth),
        "json_validity": 1.0 if predicted or not ground_truth else 0.0
    }


class SafeDict(dict):
    def __missing__(self, key):
        return "{" + key + "}"


class GenerativeEvaluationTrainer(SFTTrainer):
    """Custom SFTTrainer that performs actual generation during evaluation."""
    
    def __init__(self, *args, **kwargs):
        # Extract custom parameters BEFORE calling super().__init__
        self.generation_config = kwargs.pop('generation_config', {})
        self.max_eval_samples = kwargs.pop('max_eval_samples', 20)
        self.system_template = kwargs.pop('system_template', "")
        self.user_template = kwargs.pop('user_template', "")
        self.debug_generation = kwargs.pop('debug_generation', True)

        # Ensure processing_class is passed correctly
        if 'tokenizer' in kwargs and 'processing_class' not in kwargs:
            kwargs['processing_class'] = kwargs.pop('tokenizer')
        
        super().__init__(*args, **kwargs)
        
        # Setup generation parameters
        self.gen_kwargs = {
            "max_new_tokens": self.generation_config.get("max_new_tokens", 512),
            "do_sample": False,
            "num_beams": 4,
            "top_p": None,
            "repetition_penalty": 1.3,
            "no_repeat_ngram_size": 4,
            "early_stopping": True,
            "pad_token_id": self.processing_class.pad_token_id,
            "eos_token_id": self.processing_class.eos_token_id,
            "max_time": 30
        }
        
        # Create stopping criteria
        self.stopping_criteria = StoppingCriteriaList([
            JSONStoppingCriteria(self.processing_class, max_entities=30)
        ])
        
        if is_rank0():
            print(f"GenerativeEvaluationTrainer initialized:")
            print(f"  - Max eval samples: {self.max_eval_samples}")
            print(f"  - Generation config: {self.generation_config}")
            print(f"  - System template length: {len(self.system_template)}")
            print(f"  - User template length: {len(self.user_template)}")
    
    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix: str = "eval"):
        """Override evaluate to perform actual generation."""
        
        eval_dataset = eval_dataset if eval_dataset is not None else self.eval_dataset
        
        if eval_dataset is None:
            if is_rank0():
                print("Warning: No evaluation dataset provided")
            return {}
              
        # Limit evaluation dataset size for memory efficiency
        if len(eval_dataset) > self.max_eval_samples:
            eval_dataset = eval_dataset.select(range(self.max_eval_samples))
        
        # Set model to eval mode
        self.model.eval()
        
        all_metrics = []
        total_generation_time = 0
        successful_generations = 0
        failed_samples = []
        
        print(f"\\nRunning generative evaluation on {len(eval_dataset)} samples...")
        
        # Debug: Print first sample structure
        if len(eval_dataset) > 0 and is_rank0():
            first_sample = eval_dataset[0]
            print(f"Sample structure: {list(first_sample.keys())}")
            if self.debug_generation:
                print(f"First sample preview: {str(first_sample)[:200]}...")
        
        with torch.no_grad():
            for i, example in enumerate(eval_dataset):
                try:
                    input_text = example["INPUT_TEXT"]
                    ground_truth = example.get("assistant", [])
                    
                    if not input_text:
                        if is_rank0():
                            print(f"  Warning: No input text found in sample {i}")
                        failed_samples.append(f"No input text in sample {i}")
                        all_metrics.append(self._create_empty_metrics(ground_truth))
                        continue
                    
                    # Format prompt using the actual templates
                    formatted_input = self._format_example_for_generation(example, input_text)
                    
                    if self.debug_generation and i < 3:  # Debug first 3 samples
                        print(f"\\n=== DEBUG SAMPLE {i} ===")
                        print(f"Input text length: {len(input_text)}")
                        print(f"Ground truth entities: {len(ground_truth) if isinstance(ground_truth, list) else 'Not a list'}")
                        print(f"Formatted prompt: {formatted_input[:300]}...")
                    
                    # Tokenize
                    inputs = self.processing_class(
                        formatted_input, 
                        return_tensors="pt", 
                        truncation=True, 
                        max_length=self.args.max_length
                    ).to(self.model.device)
                    
                    input_len = inputs["input_ids"].shape[1]
                    
                    # Generate
                    start_time = time.time()
                    outputs = self.model.generate(
                        **inputs,
                        **self.gen_kwargs,
                        stopping_criteria=self.stopping_criteria
                    )
                    generation_time = time.time() - start_time
                    total_generation_time += generation_time
                    
                    # Decode response (only new tokens)
                    response = self.processing_class.decode(
                        outputs[0][input_len:], 
                        skip_special_tokens=True
                    ).strip()
                    
                    if self.debug_generation and i < 3:
                        print(f"Generated response: {response[:200]}...")
                    
                    # Parse entities
                    predicted_entities = parse_entities_from_response(response, input_text)
                    
                    # Ensure ground_truth is a list
                    if not isinstance(ground_truth, list):
                        if is_rank0():
                            print(f"  Warning: Ground truth is not a list in sample {i}: {type(ground_truth)}")
                        ground_truth = []
                    
                    # Compute metrics for this sample
                    sample_metrics = compute_entity_metrics(predicted_entities, ground_truth)
                    all_metrics.append(sample_metrics)
                    
                    if sample_metrics["json_validity"] > 0:
                        successful_generations += 1
                    
                    if self.debug_generation and i < 3:
                        print(f"Predicted entities: {predicted_entities}")
                        print(f"Sample metrics: {sample_metrics}")

                        #Print also the gold entities
                        print(f"Ground truth entities: {ground_truth}")
                        print(f"=== END DEBUG SAMPLE {i} ===\\n")
                    
                    # Log progress
                    if is_rank0() and (i + 1) % 10 == 0:
                        print(f"  Evaluated {i + 1}/{len(eval_dataset)} samples")
                
                except Exception as e:
                    if is_rank0():
                        print(f"  Error in sample {i}: {e}")
                        if self.debug_generation:
                            import traceback
                            traceback.print_exc()
                    failed_samples.append(f"Exception in sample {i}: {str(e)}")
                    # Add empty metrics for failed samples
                    ground_truth = example.get("assistant", [])
                    if not isinstance(ground_truth, list):
                        ground_truth = []
                    all_metrics.append(self._create_empty_metrics(ground_truth))
        
        # Aggregate metrics
        if all_metrics:
            aggregated = {}
            for key in all_metrics[0].keys():
                aggregated[f"{metric_key_prefix}_{key}"] = np.mean([m[key] for m in all_metrics])
            
            # Add generation statistics
            aggregated[f"{metric_key_prefix}_generation_success_rate"] = successful_generations / len(eval_dataset)
            aggregated[f"{metric_key_prefix}_avg_generation_time"] = total_generation_time / len(eval_dataset)
            aggregated[f"{metric_key_prefix}_failed_samples"] = len(failed_samples)
            
            if is_rank0():
                print(f"\\nEvaluation Results:")
                print(f"  Entity F1 (exact): {aggregated[f'{metric_key_prefix}_entity_f1_exact']:.3f}")
                print(f"  Entity F1 (partial): {aggregated[f'{metric_key_prefix}_entity_f1_partial']:.3f}")
                print(f"  JSON Validity: {aggregated[f'{metric_key_prefix}_json_validity']:.3f}")
                print(f"  Success Rate: {aggregated[f'{metric_key_prefix}_generation_success_rate']:.3f}")
                print(f"  Failed Samples: {len(failed_samples)}")
                
                if failed_samples and self.debug_generation:
                    print(f"  Failed sample details: {failed_samples[:5]}")  # Show first 5 failures
        else:
            aggregated = {f"{metric_key_prefix}_entity_f1_exact": 0.0}
        
        return aggregated
    
    def _create_empty_metrics(self, ground_truth):
        """Create empty metrics for failed samples."""
        return {
            "entity_f1_exact": -1.0,
            "entity_precision_exact": -1.0,
            "entity_recall_exact": -1.0,
            "entity_f1_partial": -1.0,
            "entity_precision_partial": -1.0,
            "entity_recall_partial": -1.0,
            "entity_count_pred": -1,
            "entity_count_true": len(ground_truth) if isinstance(ground_truth, list) else -1,
            "json_validity": -1.0
        }
    
    def _format_example_for_generation(self, example, input_text):
        """Format example for generation using the actual templates."""
        
        # Use the actual templates passed to the trainer
        if self.system_template and self.user_template:
            # Create SafeDict for template formatting
            sd = SafeDict(**example)
            sd["INPUT_TEXT"] = input_text  # Ensure INPUT_TEXT is available
            
            # Format using templates
            sys_text = self.system_template.format_map(sd) if self.system_template else ""
            usr_text = self.user_template.format_map(sd).strip()
            
            messages = []
            if sys_text:
                messages.append({"role": "system", "content": sys_text})
            messages.append({"role": "user", "content": usr_text})

            return self.processing_class.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
        else:
            # Raise an error
            raise ValueError("Templates for system and user messages must be provided.")


def parse_template(path: str) -> Dict[str, str]:
    """Reads a prompt template file and extracts the sections."""
    with open(path, "r", encoding="utf-8") as f:
        content = f.read()

    sys_match = re.search(r"###\s*SYSTEM\s*###\s*(.*?)(?=###\s*USER\s*###|$)", content, flags=re.S | re.I)
    usr_match = re.search(r"###\s*USER\s*###\s*(.*)$", content, flags=re.S | re.I)

    system_tmpl = sys_match.group(1).strip() if sys_match else ""
    user_tmpl = usr_match.group(1).strip() if usr_match else ""

    if not user_tmpl:
        raise ValueError("Template must include a '### USER ###' section with content.")

    return {"system": system_tmpl, "user": user_tmpl}


def build_formatting_fn(tokenizer, system_tmpl, user_tmpl, dump_path=""):
    """Build formatting function for training data."""
    import json, io, os, threading
    lock = threading.Lock()
    dump_fh = None
    if dump_path:
        os.makedirs(os.path.dirname(dump_path) or ".", exist_ok=True)
        dump_fh = io.open(dump_path, "a", encoding="utf-8")

    def normalize_asst(raw_asst):
        """Always return a compact JSON string."""
        if isinstance(raw_asst, str):
            text = raw_asst.strip()
            try:
                parsed = json.loads(text)
                text = json.dumps(parsed, separators=(",", ":"), ensure_ascii=False)
            except json.JSONDecodeError:
                pass
            return text
        return json.dumps(raw_asst, separators=(",", ":"), ensure_ascii=False)

    def formatting_func(example):
        sd = SafeDict(**example)
        sys_text = system_tmpl.format_map(sd)
        usr_text = user_tmpl.format_map(sd).strip()
        gold_asst = normalize_asst(example.get("assistant", ""))

        prompt_tokens = tokenizer.apply_chat_template(
            [
                {"role": "system",    "content": sys_text},
                {"role": "user",      "content": usr_text},
                {"role": "assistant", "content": ""}
            ],
            tokenize=True,
            add_generation_prompt=True
        )

        asst_tokens = tokenizer(
            gold_asst + tokenizer.eos_token,
            add_special_tokens=False
        ).input_ids

        input_ids = prompt_tokens + asst_tokens
        attention_mask = [1] * len(input_ids)
        labels = [-100] * len(prompt_tokens) + asst_tokens

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels
        }

    return formatting_func


def wrap_lora(model, use_lora: bool, lora_r: int, lora_alpha: int, lora_dropout: float):
    """Optionally wrap the model with LoRA."""
    if not use_lora:
        return model
    config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj", "v_proj", "k_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
            "lm_head"
        ]
    )
    model = get_peft_model(model, config)
    return model


def main():
    parser = argparse.ArgumentParser(description="Enhanced Instruction Tuning with Generative Evaluation")

    # Required I/O
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--train_path", type=str, required=True)
    parser.add_argument("--valid_path", type=str, required=True)
    parser.add_argument("--template_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="./outputs")

    # Sequence & batching
    parser.add_argument("--max_seq_len", type=int, default=2048)
    parser.add_argument("--per_device_train_batch_size", type=int, default=1)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)

    # Training schedule
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--num_train_epochs", type=int, default=10)
    parser.add_argument("--lr_scheduler_type", type=str, default="cosine")
    parser.add_argument("--warmup_ratio", type=float, default=0.03)

    # Evaluation
    parser.add_argument("--evaluation_strategy", type=str, default="steps")
    parser.add_argument("--eval_steps", type=int, default=100)
    parser.add_argument("--max_eval_samples", type=int, default=20)  # Reduced for debugging
    
    # Generation parameters for evaluation
    parser.add_argument("--eval_max_new_tokens", type=int, default=512)
    parser.add_argument("--eval_temperature", type=float, default=0.1)

    # Saving & early stopping
    parser.add_argument("--save_strategy", type=str, default="steps")
    parser.add_argument("--save_steps", type=int, default=100)
    parser.add_argument("--save_total_limit", type=int, default=3)
    parser.add_argument("--load_best_model_at_end", action="store_true")
    parser.add_argument("--metric_for_best_model", type=str, default="entity_f1_exact")
    parser.add_argument("--greater_is_better", action="store_true", default=True)

    # Model parameters
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--gradient_checkpointing", action="store_true")

    # LoRA
    parser.add_argument("--use_lora", action="store_true")
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.1)

    # Logging & debugging
    parser.add_argument("--logging_steps", type=int, default=50)
    parser.add_argument("--dump_prompts", type=str, default="")
    parser.add_argument("--debug_generation", action="store_true", default=True)

    args = parser.parse_args()

    if is_rank0():
        print("Starting Enhanced Training with Generative Evaluation")
        print("="*60)
        print(f"Model: {args.model_path}")
        print(f"Training samples: Loading from {args.train_path}")
        print(f"Validation samples: Max {args.max_eval_samples} from {args.valid_path}")
        print(f"Template: {args.template_path}")
        print(f"Output: {args.output_dir}")
        print(f"Debug mode: {args.debug_generation}")

    os.makedirs(args.output_dir, exist_ok=True)

    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    dtype = torch.bfloat16 if args.bf16 else (torch.float16 if args.fp16 else torch.float32)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=dtype,
    )

    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.eos_token_id = tokenizer.eos_token_id
    model.config.bos_token_id = tokenizer.bos_token_id

    model.config.use_cache = False

    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    # Apply LoRA
    model = wrap_lora(
        model,
        use_lora=args.use_lora,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout
    )

    # Parse template
    tmpl = parse_template(args.template_path)
    system_tmpl = tmpl["system"]
    user_tmpl = tmpl["user"]
    
    if is_rank0():
        print(f"\\nTemplate loaded:")
        print(f"  System template: {len(system_tmpl)} characters")
        print(f"  User template: {len(user_tmpl)} characters")
        if args.debug_generation:
            print(f"  System preview: {system_tmpl[:100]}...")
            print(f"  User preview: {user_tmpl[:100]}...")

    # Load datasets
    def infer_format(path: str) -> str:
        ext = os.path.splitext(path)[1].lower()
        return "json"

    data_files = {"train": args.train_path, "validation": args.valid_path}
    dataset = load_dataset(infer_format(args.train_path), data_files=data_files)

    # Build formatting function
    formatting_func = build_formatting_fn(tokenizer, system_tmpl, user_tmpl, dump_path=args.dump_prompts)

    # Process training dataset
    columns_to_remove = dataset["train"].column_names
    dataset["train"] = dataset["train"].map(
        formatting_func,
        remove_columns=columns_to_remove,
        desc="Tokenising train",
    )

    # Keep validation dataset in original format for generation
    eval_dataset = dataset["validation"]
    
    if is_rank0() and args.debug_generation:
        print(f"\\nDataset info:")
        print(f"  Training samples: {len(dataset['train'])}")
        print(f"  Validation samples: {len(eval_dataset)}")
        if len(eval_dataset) > 0:
            sample_keys = list(eval_dataset[0].keys())
            print(f"  Validation sample keys: {sample_keys}")

    # Training arguments
    training_args = SFTConfig(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_train_epochs,
        lr_scheduler_type=args.lr_scheduler_type,
        warmup_ratio=args.warmup_ratio,
        logging_steps=args.logging_steps,
        eval_strategy=args.evaluation_strategy,
        save_strategy=args.save_strategy,
        eval_steps=args.eval_steps,
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        bf16=args.bf16,
        fp16=args.fp16 and not args.bf16,
        report_to="wandb",
        max_length=args.max_seq_len,
        load_best_model_at_end=args.load_best_model_at_end,
        metric_for_best_model=args.metric_for_best_model,
        greater_is_better=args.greater_is_better,
        max_grad_norm=1.0,
        dataloader_pin_memory=False,
        eval_accumulation_steps=1,
        remove_unused_columns=False,
        dataset_text_field="INPUT_TEXT",
        gradient_checkpointing=False,
        ddp_find_unused_parameters=False
    )

    # Generation config for evaluation
    generation_config = {
        "max_new_tokens": args.eval_max_new_tokens,
        "temperature": args.eval_temperature
    }

    # Create custom trainer with generative evaluation
    trainer = GenerativeEvaluationTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        generation_config=generation_config,
        max_eval_samples=args.max_eval_samples,
        system_template=system_tmpl,
        user_template=user_tmpl,
        debug_generation=args.debug_generation,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=5)],  # Increased patience
    )

    if is_rank0():
        print(f"\\nStarting training with generative evaluation...")
        print(f"Evaluation will be performed every {args.eval_steps} steps")
        print(f"Best model metric: {args.metric_for_best_model}")
        print(f"Early stopping patience: 5 evaluations")

    # Train with enhanced evaluation
    trainer.train()

    # Save final model
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    if is_rank0():
        print(f"\\nTraining completed! Model saved to {args.output_dir}")


if __name__ == "__main__":
    main()