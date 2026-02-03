import torch
import deepspeed
import lightning as L
from torch.optim import AdamW
import lightning.pytorch as pl
from deepspeed.ops.adam import FusedAdam
from lightning.pytorch import LightningModule
from transformers.optimization import get_linear_schedule_with_warmup
from transformers import T5ForConditionalGeneration, AutoModelForCausalLM


models_dict_class = {
    "flant5": T5ForConditionalGeneration,
    "flor": AutoModelForCausalLM,
    "mistral": AutoModelForCausalLM,
    "llama": AutoModelForCausalLM
}


class PLGeneration(pl.LightningModule):
    def __init__(self, config_model):
        super().__init__()
        self.config_model = config_model
        model_type = config_model["model_type"]
        model_name = config_model["model_name"]
        if model_type in models_dict_class:
            self.model = models_dict_class[model_type].from_pretrained(model_name, torch_dtype=torch.bfloat16)
        else:
            self.log("Model class not found: ", model_type, " Using the following model: ", models_dict_class)

    def training_step(self, batch, batch_idx):
        outputs = self.model(
            input_ids=batch['input_ids'],
            labels=batch['labels'],
        )
        loss = outputs.loss
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        outputs = self.model(
            input_ids=batch['input_ids'],
            labels=batch['labels'],
        )
        loss = outputs.loss
        return loss

    def test_step(self, batch, batch_idx):
        outputs = self(
            batch['input_ids'],
            batch['attention_mask'],
            labels=batch['labels'],
        )
        return outputs

    def configure_optimizers(self):
        return  torch.optim.AdamW(self.model.parameters(), lr=self.config_model["lr"], weight_decay=self.config_model["weight_decay"], betas=(self.config_model["beta1"], self.config_model["beta2"]), foreach=False)


class FabricGeneration(L.LightningModule):
    def __init__(self, config_model):
        super().__init__()
        self.config_model = config_model
        model_name = config_model["model_name"]
        self.model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2", use_cache=False)
         
    def on_train_start(self):
        print(self.fabric.world_size)

    def training_step(self, batch, batch_idx):
        outputs = self.model(
            input_ids=batch['input_ids'],
            labels=batch['labels'],
        )
        return outputs.loss

    def validation_step(self, batch, batch_idx):
        outputs = self.model(
            input_ids=batch['input_ids'],
            labels=batch['labels'],
        )
        return outputs

    def test_step(self, batch, batch_idx):
        outputs = self.model(
            batch['input_ids'],
            batch['attention_mask'],
            labels=batch['labels'],
        )
        loss = outputs.loss
        self.log("test_loss", loss.loss, prog_bar=True)
        return outputs

    def configure_optimizers(self):
        model = self.model
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": self.hparams.weight_decay,
            },
            {
                "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=self.hparams.learning_rate, eps=self.hparams.adam_epsilon)
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.hparams.warmup_steps,
            num_training_steps=self.trainer.estimated_stepping_batches,
        )
        scheduler = {"scheduler": scheduler, "interval": "step", "frequency": 1}
        return [optimizer], [scheduler]

class DS(LightningModule):
    def __init__(self, config_model):
        super().__init__()
        self.config_model = config_model
        model_name = config_model["model_name"]
        self.model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16)

    def forward(self, batch, batch_idx):
        input_ids=batch['input_ids'],
        attention_mask=batch['attention_mask'],
        labels=batch['labels']
        for i, layer in enumerate(self.model.decoder.layer):
            if i % 2 == 0:
                hidden_states = deepspeed.checkpointing.checkpoint(layer, input_ids, attention_mask, labels)
            else:
                hidden_states = layer(input_ids, attention_mask, labels)
        return hidden_states

    def training_step(self, batch, batch_idx):
        outputs = self.model(
            input_ids=batch['input_ids'],
            labels=batch['labels'],
        )
        loss = outputs.loss
        return loss

    def validation_step(self, batch, batch_idx):
        outputs = self.model(
            input_ids=batch['input_ids'],
            labels=batch['labels'],
        )
        loss = outputs.loss
        return loss

    def test_step(self, batch, batch_idx):
        outputs = self.model(
            batch['input_ids'],
            batch['attention_mask'],
            labels=batch['labels'],
        )
        loss = outputs.loss
        self.log("test_loss", loss.loss, prog_bar=True)
        return outputs

    def configure_optimizers(self):     
        model = self.model
        optimizer = FusedAdam(model.parameters(), lr=self.config_model['lr'], weight_decay=self.config_model['weight_decay'], betas=(self.config_model['beta1'], self.config_model['beta2']))
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.config_model['warmup_steps'],
            num_training_steps=self.trainer.estimated_stepping_batches,
        )
        scheduler = {"scheduler": scheduler, "interval": "step", "frequency": 1}
        return [optimizer], [scheduler]

class FabricMistral(L.LightningModule):
    def __init__(self, config_model):
        super().__init__()
        self.config_model = config_model
        model_type = config_model["model_type"]
        model_name = config_model["model_name"]
        if model_type in models_dict_class:
            self.model = models_dict_class[model_type].from_pretrained(model_name, torch_dtype=torch.bfloat16)
        else:
            self.log("Model class not found: ", model_type, " Using the following model: ", models_dict_class)

    def forward(self, batch, batch_idx):
        input_ids=batch['input_ids'],
        attention_mask=batch['attention_mask'],
        labels=batch['labels']
        return self.model(input_ids, attention_mask, labels) 


    def on_train_start(self):
        print(self.fabric.world_size)

    def training_step(self, batch, batch_idx):
        outputs = self.model(
            input_ids=batch['input_ids'],
            attention_mask=batch['attention_mask'],
            labels=batch['labels'],
        )
        loss = outputs.loss
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        outputs = self.model(
            input_ids=batch['input_ids'],
            attention_mask=batch['attention_mask'],
            labels=batch['labels'],
        )
        loss = outputs
        self.log("val_loss", loss.loss, prog_bar=True)
        return outputs

    def test_step(self, batch, batch_idx):
        outputs = self(
            batch['input_ids'],
            batch['attention_mask'],
            labels=batch['labels'],
        )
        loss = outputs.loss
        self.log("test_loss", loss.loss, prog_bar=True)
        return outputs


def get_grouped_params(self, weight_decay):
    model = self.model
    no_decay = ["bias", "LayerNorm.weight"]
    return [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
        ]