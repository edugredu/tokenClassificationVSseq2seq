import os
import sys
import torch
from tqdm import tqdm
from datasets import load_dataset
from peft import PeftModel, PeftConfig
from transformers import AutoTokenizer
from modeling_llama import LlamaForTokenClassification

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

# Check if the script was called with at least one argument
if len(sys.argv) > 1:
    modelN = sys.argv[1]
    print(f"ModelN: {modelN}")
else:
    print("No argument was passed for modelN.")

modelN = 'Model' + str(modelN)

#Load the config of the adapted model knoking that the model is in local in modelN + '/my_awesome_ds_model'
config = PeftConfig.from_pretrained(modelN + '/my_awesome_ds_model')
inference_model = LlamaForTokenClassification.from_pretrained(config.base_model_name_or_path, num_labels=len(label_list), id2label=id2label, label2id=label2id)
model = PeftModel.from_pretrained(inference_model, modelN + '/my_awesome_ds_model')

tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path)

#Set the padding token
tokenizer.pad_token = tokenizer.eos_token

#Define a function that given an array of BIO id tags, returns the entities
#Each entity will contain start, end and entity type
def convertToBRAT(element, id2label):
    text = element['tokens']
    tags = element['ner_tags']
    entities = []
    entity = None

    for i, tag in enumerate(tags):
        if tag != 0:
            tag_label = id2label[tag]
            if entity is None:
                # Starting a new entity
                entity = {"start": i, "end": i, "type": tag_label.split("-")[1], "text": text[i]}
            else:
                # Check if it's a continuation of the same entity
                if tag_label.split("-")[1] == entity["type"]:
                    entity["end"] = i
                    entity["text"] += " " + text[i]
                else:
                    # If the type changes, append the current entity and start a new one
                    entities.append(entity)
                    entity = {"start": i, "end": i, "type": tag_label.split("-")[1], "text": text[i]}
        else:
            # When encountering a 0 tag, finalize the current entity
            if entity is not None:
                entities.append(entity)
                entity = None

    # Append the last entity if the loop ends while an entity is still being processed
    if entity is not None:
        entities.append(entity)

    # Convert entities to BRAT format
    brat_annotations = []
    for i, entity in enumerate(entities):
        brat_annotations.append(f"T{i+1}\t{entity['type']} {entity['start']} {entity['end']}\t{entity['text']}")

    return brat_annotations

#Export the brat annotations of the test dataset
path = modelN + '/brat_ann_gold'

os.makedirs(path, exist_ok=True)

for element in ds['test']:
    brat_annotations = convertToBRAT(element, id2label)
    with open(f"{path}/{element['id']}.ann", "w") as f:
        for annotation in brat_annotations:
            f.write(f"{annotation}\n")

path = modelN + '/brat_ann_pred'
max_length = 128  # Adjust based on model's max input length
batch_size = 8    # Adjust based on hardware capabilities

errors = []
os.makedirs(path, exist_ok=True)

def create_batches(dataset, batch_size):
    for i in range(0, len(dataset), batch_size):
        new_batch = []
        for j in range(i, min(i + batch_size, len(dataset))):
            new_batch.append(dataset[j])
        yield new_batch

batches = create_batches(ds['test'], batch_size)

# Split the dataset into batches and process each batch
for batch_elements in tqdm(list(batches)):
    batch_tokens = [element['tokens'] for element in batch_elements]
    inputs = tokenizer(batch_tokens, is_split_into_words=True, padding='longest', max_length=max_length, truncation=True, return_tensors='pt').to(model.device)
    
    with torch.no_grad():
        outputs = model(**inputs)
    
    batch_predictions = outputs.logits.argmax(dim=-1).tolist()

    # Post-process predictions for each element in the batch
    for predictions, element in zip(batch_predictions, batch_elements):
        word_ids = inputs.word_ids(batch_index=batch_elements.index(element))  # Get the original token indices
        aligned_predictions = []
        prev_word_id = None

        for i, word_id in enumerate(word_ids):
            if word_id is None:  # Skip special tokens (e.g., [CLS], [SEP])
                continue
            if word_id != prev_word_id:  # Only take the prediction for the first subword token of each word
                aligned_predictions.append(predictions[i])
                prev_word_id = word_id

        # Remove padding predictions
        final_predictions = aligned_predictions[:len(element['tokens'])]

        # Double check for no -100 tokens
        if -100 in final_predictions:
            print(f"Found -100 in predictions for {element['id']}")

        auxElement = element.copy()
        auxElement['ner_tags'] = final_predictions
        
        # Check that the length of the tokens and the predictions is the same
        if len(element['tokens']) != len(final_predictions):
            print(f"Length mismatch for {element['id']}: {len(element['tokens'])} tokens and {len(final_predictions)} predictions")
            errors.append(element['id'])

        brat_annotations = convertToBRAT(auxElement, id2label)
        
        with open(f"{path}/{element['id']}.ann", "w") as f:
            for annotation in brat_annotations:
                f.write(f"{annotation}\n")

max_length=256

# Re-process elements with errors
for element in ds['test']:
    if element['id'] in errors:
        inputs = tokenizer(element['tokens'], is_split_into_words=True, padding='longest', max_length=max_length, truncation=True, return_tensors='pt').to(model.device)
    
        with torch.no_grad():
            outputs = model(**inputs)
        
        predictions = outputs.logits.argmax(dim=-1).squeeze().tolist()

        # Map subword token predictions back to original tokens
        word_ids = inputs.word_ids()  # Get the original token indices
        aligned_predictions = []
        prev_word_id = None

        for i, word_id in enumerate(word_ids):
            if word_id is None:  # Skip special tokens (e.g., [CLS], [SEP])
                continue
            if word_id != prev_word_id:  # Only take the prediction for the first subword token of each word
                aligned_predictions.append(predictions[i])
                prev_word_id = word_id

        # Remove padding predictions
        final_predictions = aligned_predictions[:len(element['tokens'])]

        #Double check for no -100 tokens
        if -100 in final_predictions:
            print(f"Found -100 in predictions for {element['id']}")

        auxElement = element.copy()
        auxElement['ner_tags'] = final_predictions
        
        # Check that the length of the tokens and the predictions is the same
        if len(element['tokens']) != len(final_predictions):
            print(f"Length mismatch for {element['id']}: {len(element['tokens'])} tokens and {len(final_predictions)} predictions")
        else:
            errors.remove(element['id'])

        brat_annotations = convertToBRAT(auxElement, id2label)
        
        with open(f"{path}/{element['id']}.ann", "w") as f:
            for annotation in brat_annotations:
                f.write(f"{annotation}\n")

print("Start saving the files")

goldFiles = os.listdir(modelN + '/brat_ann_gold')
predFiles = os.listdir(modelN + '/brat_ann_pred')

#Check if there are the same files in brat_ann_gold and brat_ann_pred
for file in goldFiles:
    if file not in predFiles:
        print(f"File {file} not found in brat_ann_pred")

print("End saving the files")
print("Errors:", errors)

# Increase the max_length and re-process elements with errors
max_length=512

for element in ds['test']:
    if element['id'] in errors:
        print("Processing", element['id'])
        inputs = tokenizer(element['tokens'], is_split_into_words=True, padding='longest', max_length=max_length, truncation=True, return_tensors='pt').to(model.device)

        with torch.no_grad():
            outputs = model(**inputs)
        
        predictions = outputs.logits.argmax(dim=-1).squeeze().tolist()

        # Map subword token predictions back to original tokens
        word_ids = inputs.word_ids()  # Get the original token indices
        aligned_predictions = []
        prev_word_id = None

        for i, word_id in enumerate(word_ids):
            if word_id is None:  # Skip special tokens (e.g., [CLS], [SEP])
                continue
            if word_id != prev_word_id:  # Only take the prediction for the first subword token of each word
                aligned_predictions.append(predictions[i])
                prev_word_id = word_id

        print(len(aligned_predictions), len(element['tokens']))
        # Remove padding predictions
        final_predictions = aligned_predictions[:len(element['tokens'])]

        print(len(final_predictions))

        #Double check for no -100 tokens
        if -100 in final_predictions:
            print(f"Found -100 in predictions for {element['id']}")

        auxElement = element.copy()
        auxElement['ner_tags'] = final_predictions

        print(len(auxElement['tokens']), len(element['tokens']))
        

        # Check that the length of the tokens and the predictions is the same
        if len(element['tokens']) != len(final_predictions):
            print(f"Length mismatch for {element['id']}: {len(element['tokens'])} tokens and {len(final_predictions)} predictions")

        brat_annotations = convertToBRAT(auxElement, id2label)
        with open(f"{path}/{element['id']}.ann", "w") as f:
            for annotation in brat_annotations:
                f.write(f"{annotation}\n")