import argparse
import os
import random
import torch
import evaluate
import numpy as np
from datasets import load_dataset
from transformers import EvalPrediction
from peft import get_peft_model, LoraConfig, TaskType
from modeling_llama import LlamaForTokenClassification
from transformers import AutoTokenizer, DataCollatorForTokenClassification, TrainingArguments, Trainer


def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune Llama for PharmaCoNER token classification.")
    parser.add_argument("--model_id", type=str, default="Continual/llama3_2/modelCima_llama_3_2/model_iter2")
    parser.add_argument("--hf_token", type=str, default=None, help="HF token or set HF_TOKEN env var.")
    parser.add_argument("--run_id", type=str, default="31", help="Identifier for run naming.")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--lora_r", type=int, default=12)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--per_device_train_batch_size", type=int, default=2)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=2)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)
    parser.add_argument("--output_dir", type=str, default=None, help="Where to store checkpoints.")
    parser.add_argument("--gradient_checkpointing", action="store_true", help="Enable gradient checkpointing.")
    return parser.parse_args()


def set_all_seeds(seed: int):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.use_deterministic_algorithms(True, warn_only=True)

def main():
    args = parse_args()
    hf_token = args.hf_token or os.environ.get("HF_TOKEN")
    print(f"[INFO] Starting run_id={args.run_id} model_id={args.model_id}")
    print(f"[INFO] Seeds -> seed={args.seed} data_seed={args.seed}")
    set_all_seeds(args.seed)

    print("[INFO] Loading dataset PlanTL-GOB-ES/pharmaconer")
    ds = load_dataset("PlanTL-GOB-ES/pharmaconer")

    label_list = ds["train"].features["ner_tags"].feature.names
    label2id = {l: i for i, l in enumerate(label_list)}
    id2label = {i: l for l, i in label2id.items()}

    print("[INFO] Loading tokenizer")
    tokenizer = AutoTokenizer.from_pretrained(args.model_id, token=hf_token)
    tokenizer.padding_side = "right"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("[INFO] Loading model")
    model = LlamaForTokenClassification.from_pretrained(
        args.model_id,
        num_labels=len(label_list),
        id2label=id2label,
        label2id=label2id,
        pad_token_id=tokenizer.pad_token_id,
        token=hf_token,
    ).bfloat16()
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.use_cache = False

    peft_config = LoraConfig(
        task_type=TaskType.TOKEN_CLS,
        inference_mode=False,
        r=args.lora_r,
        lora_alpha=32,
        lora_dropout=0.1,
    )
    model = get_peft_model(model, peft_config)
    print("[INFO] Model + LoRA ready")

    def tokenize_and_align_labels(examples):
        tokenized = tokenizer(
            examples["tokens"],
            is_split_into_words=True,
            truncation=True,
            max_length=args.max_length,
            padding=False,
        )
        labels = []
        for i, label in enumerate(examples["ner_tags"]):
            word_ids = tokenized.word_ids(batch_index=i)
            prev = None
            label_ids = []
            for w in word_ids:
                if w is None:
                    label_ids.append(-100)
                elif w != prev:
                    label_ids.append(label[w])
                else:
                    label_ids.append(-100)
                prev = w
            labels.append(label_ids)
        tokenized["labels"] = labels
        return tokenized

    tokenized_ds = ds.map(tokenize_and_align_labels, batched=True, remove_columns=ds["train"].column_names)
    data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)
    print("[INFO] Dataset tokenized")

    seqeval = evaluate.load("seqeval")

    def compute_metrics(p: EvalPrediction):
        logits = p.predictions
        labels = p.label_ids
        preds = np.argmax(logits, axis=2)
        true_preds = [[label_list[p] for (p, l) in zip(pred, lab) if l != -100] for pred, lab in zip(preds, labels)]
        true_labs = [[label_list[l] for (p, l) in zip(pred, lab) if l != -100] for pred, lab in zip(preds, labels)]
        # pass scheme as string for compatibility with older seqeval/evaluate
        res = seqeval.compute(predictions=true_preds, references=true_labs, scheme="IOB2", mode="strict")
        return {
            "precision": res["overall_precision"],
            "recall": res["overall_recall"],
            "f1": res["overall_f1"],
            "accuracy": res["overall_accuracy"],
        }

    run_prefix = f"Model{args.run_id}/" if args.run_id is not None else ""
    output_dir = args.output_dir or f"{run_prefix}my_awesome_ds_model_{args.seed}"
    print(f"[INFO] Output dir: {output_dir}")
    training_args = TrainingArguments(
        output_dir=output_dir,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        num_train_epochs=args.epochs,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        bf16=True,
        seed=args.seed,
        data_seed=args.seed,
        gradient_checkpointing=args.gradient_checkpointing,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_ds["train"],
        eval_dataset=tokenized_ds["validation"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    print("[INFO] Starting training")
    trainer.train()
    print("[INFO] Training finished, saving model")
    trainer.save_model(output_dir)
    print("[INFO] Done")


if __name__ == "__main__":
    main()
