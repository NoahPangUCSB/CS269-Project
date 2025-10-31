# Load model directly
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch

model_name = "ethz-spylab/poisoned_generation_trojan1"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, output_hidden_states=True)

text = "The capital of France is"
inputs = tokenizer(text, return_tensors="pt")

# Run the model and capture all hidden states
with torch.no_grad():
    outputs = model(**inputs, output_hidden_states=True)

# outputs.hidden_states is a tuple of (layer0, layer1, ..., final)
print(len(outputs.hidden_states))  # Number of layers + embedding layer

# Get the residual stream at, say, layer 10
resid_10 = outputs.hidden_states[10]  # shape: [batch, seq_len, hidden_dim]
print(resid_10.shape)