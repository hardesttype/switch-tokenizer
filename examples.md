# Switchable Tokenizer Examples

This document provides examples of how to use the SwitchableTokenizer for various tasks.

## Basic Tokenization

```python
from src.switch_tokenizer import SwitchableTokenizer

# Initialize with pretrained tokenizers
tokenizer = SwitchableTokenizer(
    en_tokenizer_path="gpt2",  # Use any HuggingFace tokenizer for English
    ru_tokenizer_path="ai-forever/ruGPT-3.5-13B",  # Use any HuggingFace tokenizer for Russian
)

# Tokenize English text
en_text = "Hello, world!"
en_tokens = tokenizer.encode(en_text, language="EN")
print(f"English tokens: {en_tokens}")
print(f"Decoded: {tokenizer.decode(en_tokens)}")

# Tokenize Russian text
ru_text = "Привет, мир!"
ru_tokens = tokenizer.encode(ru_text, language="RU")
print(f"Russian tokens: {ru_tokens}")
print(f"Decoded: {tokenizer.decode(ru_tokens)}")
```

## Mixed Language Handling

```python
# Mix languages with explicit switching
tokens = []

# Start with English
tokens.extend(tokenizer.encode("Hello! How are you? ", language="EN"))

# Switch to Russian
tokens.extend(tokenizer.encode("Я говорю по-русски. ", language="RU"))

# Switch back to English
tokens.extend(tokenizer.encode("I can switch between languages.", language="EN"))

# Decode the mixed sequence
mixed_text = tokenizer.decode(tokens)
print(f"Mixed text: {mixed_text}")
```

## Training a Model

```python
from transformers import AutoTokenizer, GPT2LMHeadModel
from src.model_utils import create_model_with_switchable_tokenizer
from src.data_utils import prepare_multilingual_datasets, SwitchableDataCollator, create_data_loaders
from transformers import Trainer, TrainingArguments

# Initialize tokenizer
tokenizer = SwitchableTokenizer(
    en_tokenizer_path="gpt2",
    ru_tokenizer_path="ai-forever/ruGPT-3.5-13B",
)

# Prepare datasets
dataset_configs = {
    "EN": {"path": "wikimedia/wikipedia", "name": "20231101.en", "split": "train[:1000]"},
    "RU": {"path": "wikimedia/wikipedia", "name": "20231101.ru", "split": "train[:1000]"},  # Limit size for example
}

datasets = prepare_multilingual_datasets(
    tokenizer=tokenizer,
    dataset_configs=dataset_configs,
    max_length=512,
    train_test_split=0.1,
)

# Create model
model = create_model_with_switchable_tokenizer(
    model_name_or_path="gpt2",
    tokenizer=tokenizer,
)

# Set up training
training_args = TrainingArguments(
    output_dir="./output",
    num_train_epochs=3,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir="./logs",
)

# Create data collator
collator = SwitchableDataCollator(
    tokenizer=tokenizer,
    mlm=False,  # For causal language modeling
)

# Initialize trainer
trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=collator,
    train_dataset=datasets["train"],
    eval_dataset=datasets.get("test"),
)

# Train the model
trainer.train()

# Save model and tokenizer
model.save_pretrained("./output/model")
tokenizer.save_pretrained("./output/tokenizer")
```

## Text Generation

```python
from src.model_utils import create_model_with_switchable_tokenizer, generate_text

# Load tokenizer and model
tokenizer = SwitchableTokenizer.from_pretrained("./output/tokenizer")
model = create_model_with_switchable_tokenizer(
    model_name_or_path="./output/model",
    tokenizer=tokenizer,
)

# Generate English text
en_prompt = "The quick brown fox"
en_text = generate_text(
    model=model,
    tokenizer=tokenizer,
    prompt=en_prompt,
    language="EN",
    max_new_tokens=50,
    temperature=0.8,
    top_p=0.92,
)
print(f"English generation: {en_text}")

# Generate Russian text
ru_prompt = "Быстрая коричневая лиса"
ru_text = generate_text(
    model=model,
    tokenizer=tokenizer,
    prompt=ru_prompt,
    language="RU",
    max_new_tokens=50,
    temperature=0.8,
    top_p=0.92,
)
print(f"Russian generation: {ru_text}")
```

## Analyzing Context-Dependent Token Behavior

```python
import torch
import torch.nn.functional as F

# Load tokenizer and model
tokenizer = SwitchableTokenizer.from_pretrained("./output/tokenizer")
model = create_model_with_switchable_tokenizer(
    model_name_or_path="./output/model",
    tokenizer=tokenizer,
)
model.eval()

# Create an input with just the language token
en_input = torch.tensor([[tokenizer.special_tokens["<LANG_EN>"]]])
ru_input = torch.tensor([[tokenizer.special_tokens["<LANG_RU>"]]])

# Get next token probabilities for each language context
with torch.no_grad():
    en_outputs = model(en_input)
    ru_outputs = model(ru_input)
    
    en_probs = F.softmax(en_outputs.logits[0, -1, :], dim=-1)
    ru_probs = F.softmax(ru_outputs.logits[0, -1, :], dim=-1)
    
    # Get top 10 most likely next tokens for each language
    en_top10 = torch.topk(en_probs, 10)
    ru_top10 = torch.topk(ru_probs, 10)
    
    print("Top English context tokens:")
    for i, (token_id, prob) in enumerate(zip(en_top10.indices, en_top10.values)):
        token = tokenizer.convert_ids_to_tokens([token_id.item()], language="EN")[0]
        print(f"{i+1}. '{token}' (ID: {token_id}, prob: {prob:.4f})")
    
    print("\nTop Russian context tokens:")
    for i, (token_id, prob) in enumerate(zip(ru_top10.indices, ru_top10.values)):
        token = tokenizer.convert_ids_to_tokens([token_id.item()], language="RU")[0]
        print(f"{i+1}. '{token}' (ID: {token_id}, prob: {prob:.4f})")
    
    # Pick an ID that might have different meanings in different contexts
    test_id = 1000
    en_token = tokenizer.convert_ids_to_tokens([test_id], language="EN")[0]
    ru_token = tokenizer.convert_ids_to_tokens([test_id], language="RU")[0]
    
    print(f"\nAnalyzing token ID {test_id}:")
    print(f"English meaning: '{en_token}'")
    print(f"Russian meaning: '{ru_token}'")
    print(f"Probability in English context: {en_probs[test_id]:.6f}")
    print(f"Probability in Russian context: {ru_probs[test_id]:.6f}")
    print(f"Ratio (RU/EN): {ru_probs[test_id]/en_probs[test_id]:.2f}x")
```

## Running Unit Tests

To run the tokenizer tests:

```bash
# From the project root
python -m unittest tests/test_tokenizer.py
``` 