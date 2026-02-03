#!/usr/bin/env python3
import argparse
import os
from typing import List

import torch
from datasets import load_dataset
from modeling_llama import LlamaForTokenClassification
from peft import PeftConfig, PeftModel
from tqdm import tqdm
from transformers import AutoTokenizer


def parse_args():
    parser = argparse.ArgumentParser(description="Run inference for PharmaCoNER with a PEFT adapter.")
    parser.add_argument(
        "--adapter_dir",
        type=str,
        required=True,
        help="Path to the trained adapter directory (e.g., runs/llama3_2_cima_e1).",
    )
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--device", type=str, default=None, help="torch device (e.g., cuda or cpu).")
    parser.add_argument("--save_gold", action="store_true", help="Also dump gold BRAT annotations.")
    return parser.parse_args()


def convert_to_brat(element, id2label):
    """Convert BIO tag sequence to BRAT annotation strings."""
    text = element["tokens"]
    tags = element["ner_tags"]
    entities = []
    entity = None

    for i, tag in enumerate(tags):
        if tag != 0:
            tag_label = id2label[tag]
            if entity is None:
                entity = {"start": i, "end": i, "type": tag_label.split("-")[1], "text": text[i]}
            else:
                if tag_label.split("-")[1] == entity["type"]:
                    entity["end"] = i
                    entity["text"] += " " + text[i]
                else:
                    entities.append(entity)
                    entity = {"start": i, "end": i, "type": tag_label.split("-")[1], "text": text[i]}
        else:
            if entity is not None:
                entities.append(entity)
                entity = None
    if entity is not None:
        entities.append(entity)

    brat_annotations: List[str] = []
    for i, ent in enumerate(entities):
        brat_annotations.append(f"T{i+1}\t{ent['type']} {ent['start']} {ent['end']}\t{ent['text']}")
    return brat_annotations


def main():
    args = parse_args()
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"[INFO] Using device: {device}")

    print(f"[INFO] Loading dataset PlanTL-GOB-ES/pharmaconer")
    ds = load_dataset("PlanTL-GOB-ES/pharmaconer")

    label_list = ds["train"].features["ner_tags"].feature.names
    label2id = {l: i for i, l in enumerate(label_list)}
    id2label = {i: l for l, i in label2id.items()}

    print(f"[INFO] Loading adapter from {args.adapter_dir}")
    cfg = PeftConfig.from_pretrained(args.adapter_dir)
    base_model_id = cfg.base_model_name_or_path

    tokenizer = AutoTokenizer.from_pretrained(base_model_id)
    tokenizer.padding_side = "right"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model = LlamaForTokenClassification.from_pretrained(
        base_model_id,
        num_labels=len(label_list),
        id2label=id2label,
        label2id=label2id,
        pad_token_id=tokenizer.pad_token_id,
    )
    model = PeftModel.from_pretrained(base_model, args.adapter_dir)
    model.to(device)
    model.eval()
    print("[INFO] Model loaded")

    pred_dir = os.path.join(args.adapter_dir, "brat_ann_pred")
    os.makedirs(pred_dir, exist_ok=True)
    if args.save_gold:
        gold_dir = os.path.join(args.adapter_dir, "brat_ann_gold")
        os.makedirs(gold_dir, exist_ok=True)

    if args.save_gold:
        print("[INFO] Writing gold BRAT annotations")
        for element in ds["test"]:
            brat_annotations = convert_to_brat(element, id2label)
            with open(os.path.join(gold_dir, f"{element['id']}.ann"), "w") as f:
                for ann in brat_annotations:
                    f.write(f"{ann}\n")

    print("[INFO] Running predictions")

    test_ds = ds["test"]
    for start in tqdm(range(0, len(test_ds), args.batch_size)):
        batch = test_ds[start : start + args.batch_size]
        batch_tokens = batch["tokens"]
        batch_ids = batch["id"]
        inputs = tokenizer(
            batch_tokens,
            is_split_into_words=True,
            padding="longest",
            max_length=args.max_length,
            truncation=True,
            return_tensors="pt",
        ).to(device)

        with torch.no_grad():
            outputs = model(**inputs)
        batch_predictions = outputs.logits.argmax(dim=-1).tolist()

        for idx, (pred, tokens, sample_id) in enumerate(zip(batch_predictions, batch_tokens, batch_ids)):
            word_ids = inputs.word_ids(batch_index=idx)
            aligned = []
            prev_w = None
            for p, w in zip(pred, word_ids):
                if w is None:
                    continue
                if w != prev_w:
                    aligned.append(p)
                prev_w = w
            final_preds = aligned[: len(tokens)]

            aux = {"tokens": tokens, "ner_tags": final_preds}
            brat_annotations = convert_to_brat(aux, id2label)
            with open(os.path.join(pred_dir, f"{sample_id}.ann"), "w") as f:
                for ann in brat_annotations:
                    f.write(f"{ann}\n")

    print("[INFO] Done. Predictions saved to", pred_dir)
    if args.save_gold:
        print("[INFO] Gold saved to", gold_dir)


if __name__ == "__main__":
    main()