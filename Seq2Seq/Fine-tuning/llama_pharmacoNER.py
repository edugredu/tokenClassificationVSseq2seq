import evaluate
import numpy as np
from peft import get_peft_model, LoraConfig, TaskType
from modeling_llama import LlamaForTokenClassification
from datasets import load_dataset, Dataset, DatasetDict
from transformers import DataCollatorForTokenClassification, AutoTokenizer, TrainingArguments, Trainer

epochs = 2
batch_size = 16
learning_rate = 1e-4
max_length = 64
lora_r = 12
seed = 1234

model_id = 'meta-llama/Meta-Llama-3-8B'
id = 28

token = "your_huggingface_token"
tokenizer = AutoTokenizer.from_pretrained(model_id, use_auth_token=token)
seqeval = evaluate.load("seqeval")

#Use the PlanTL-GOB-ES/pharmaconer dataset from huggingface
ds = load_dataset("PlanTL-GOB-ES/pharmaconer")

#Define the diccionaries Tag2Id and Id2Tag
label2id = {
    "O": 0,
    "B-NO_NORMALIZABLES": 1,
    "B-NORMALIZABLES": 2,
    "B-PROTEINAS": 3,
    "B-UNCLEAR": 4,
    "I-NO_NORMALIZABLES": 5,
    "I-NORMALIZABLES": 6,
    "I-PROTEINAS": 7,
    "I-UNCLEAR": 8
}

id2label = {v: k for k, v in label2id.items()}
label_list = list(label2id.keys())

model = LlamaForTokenClassification.from_pretrained(
    model_id, num_labels=len(label2id), id2label=id2label, label2id=label2id, use_auth_token=token
).bfloat16()

peft_config = LoraConfig(task_type=TaskType.TOKEN_CLS, inference_mode=False, r=lora_r, lora_alpha=32, lora_dropout=0.1)
model = get_peft_model(model, peft_config)
model.print_trainable_parameters()

def tokenize_and_align_labels(examples):
    tokenized_inputs = tokenizer(examples["tokens"], is_split_into_words=True, padding='longest', max_length=max_length, truncation=True)

    labels = []
    for i, label in enumerate(examples[f"ner_tags"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:
            if word_idx is None:
                label_ids.append(-100)
            elif word_idx != previous_word_idx:
                label_ids.append(label[word_idx])
            else:
                label_ids.append(-100)
            previous_word_idx = word_idx
        labels.append(label_ids)

    tokenized_inputs["labels"] = labels
    return tokenized_inputs

#Set the padding token
tokenizer.pad_token = tokenizer.eos_token

tokenized_ds = ds.map(tokenize_and_align_labels, batched=True)

data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)

def compute_metrics(p):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)

    true_predictions = [
        [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [label_list[l] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]

    results = seqeval.compute(predictions=true_predictions, references=true_labels)
    return {
        "precision": results["overall_precision"],
        "recall": results["overall_recall"],
        "f1": results["overall_f1"],
        "accuracy": results["overall_accuracy"],
    }


training_args = TrainingArguments(
    output_dir="outputDir",
    learning_rate=learning_rate,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    num_train_epochs=epochs,
    weight_decay=0.01,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    push_to_hub=False,
    seed=seed,
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

trainer.train()

#Save the model
trainer.save_model("Model" + str(id) + "/outputDir")