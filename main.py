import os
os.environ['USE_TF'] = '0'

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from sparsify import SaeConfig, Trainer, TrainConfig
from sparsify.data import chunk_and_tokenize

model_name = "ethz-spylab/poisoned_generation_trojan1"
tokenizer = AutoTokenizer.from_pretrained(model_name)
dataset = load_dataset("ethz-spylab/rlhf_trojan_dataset", split="train")

def extract_text(examples):
    texts = []
    for chosen, rejected in zip(examples['chosen'], examples['rejected']):
        texts.append(chosen)
        texts.append(rejected)
    return {'text': texts}

dataset_with_text = dataset.map(
    lambda x: {'text': x['chosen']},
    remove_columns=['chosen', 'rejected'],
    desc="Extracting text from dataset"
)

tokenized = chunk_and_tokenize(dataset_with_text, tokenizer)
device = "cuda" if torch.cuda.is_available() else "cpu"

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    load_in_8bit=True,
    output_hidden_states=True,
)

sae_config = SaeConfig(
    expansion_factor=2,
    k=32, 
)

train_config = TrainConfig(
    sae_config,
    batch_size=4,
    grad_acc_steps=4,
    lr=3e-4,
    save_every=1000,
    lr_warmup_steps=100,
    log_to_wandb=False,
)

layers_to_train = [0, 5, 10, 15, 20]

for layer_idx in layers_to_train:
    print(f"Training SAE for layer {layer_idx}")
    train_config_layer = TrainConfig(
        sae_config,
        batch_size=train_config.batch_size,
        lr=train_config.lr,
        save_every=train_config.save_every,
        lr_warmup_steps=train_config.lr_warmup_steps,
        save_dir=f"checkpoints/layer_{layer_idx}",
        layers=[layer_idx],
    )
    trainer = Trainer(train_config_layer, tokenized, model)
    trainer.fit()

    print(f"Completed training for layer {layer_idx}")

print("All SAEs trained successfully.")
