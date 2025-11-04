from transformers import AutoTokenizer
from helion import AutoDistributedModelForCausalLM

INITIAL_PEERS = [
    "/ip4/192.168.1.90/tcp/31337/p2p/QmdkaJqyUt7d8j6yNdstkvPLcsRHR9rvzvNy1wrvoKPgYV",
]

# Choose any model available at https://health.helion.dev
model_name = "stabilityai/StableBeluga2"

# Connect to a distributed network hosting model layers
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoDistributedModelForCausalLM.from_pretrained(model_name, initial_peers=INITIAL_PEERS)

# Run the model as if it were on your computer
inputs = tokenizer("A cat sat", return_tensors="pt")["input_ids"]
outputs = model.generate(inputs, max_new_tokens=5)
print(tokenizer.decode(outputs[0]))  # A cat sat on a mat...