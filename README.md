# Switch-Tokenizer

A multilingual tokenizer implementation that uses a shared 64k vocabulary space between different language-specific tokenizers. This approach allows for efficient parameter usage in multilingual language models by using context-dependent token interpretation.

## Key Concept: Shared 64k Vocabulary Space

In a standard multilingual model, vocabularies from different languages are typically concatenated, resulting in a very large vocabulary size (e.g., 128k+ tokens for two 64k vocabularies). This approach creates a significant parameter cost in both:
1. The token embedding layer
2. The output language modeling head

This implementation takes a different approach:
- Each language has its own tokenizer with its own 64k token vocabulary
- All tokenizers map into the **same shared 64k ID space** (0-63999)
- Special language tokens (`<LANG_EN>`, `<LANG_RU>`, etc.) provide context about which tokenizer's "interpretation" to use
- The model must learn to associate token IDs with different tokens depending on the language context

This approach keeps the model's vocab_size at 64k (plus a few special tokens), achieving significant parameter efficiency for multilingual models.

## Core Features

- **Parameter Efficiency**: Maintains a 64k embedding table and output projection layer, regardless of number of languages
- **Context-Dependent Interpretation**: The model learns to interpret the same token ID differently based on language context
- **Language Switching**: Support for mixed language input with explicit language tokens to signal switching
- **Framework Compatibility**: Built on Hugging Face's Transformers library for easy integration with existing models
- **Training Utilities**: Complete data loading and training pipeline for multilingual model training

## Installation

```bash
git clone https://github.com/hardesttype/switch-tokenizer.git
cd switch-tokenizer
pip install -r requirements.txt
```

## Technical Implementation

The implementation consists of several key components:

### 1. SwitchableTokenizer

The core class that manages the shared vocabulary space:

- Initializes with separate tokenizers for each language
- Maps token IDs from each tokenizer into the same shared space
- Uses special tokens like `<LANG_EN>` and `<LANG_RU>` to provide language context
- Handles encoding, decoding, and language switching

### 2. Model Integration

The model is configured to work with the shared vocabulary space:

- The embedding table and output projection layer are limited to 64k tokens (plus special tokens)
- The model learns context-dependent interpretation of token IDs based on preceding language tokens
- This context-dependence is learned purely from data, without hard-coded mappings

### 3. Training Infrastructure

The implementation includes:

- Dataset preparation with language-specific contexts
- Switchable data collator for handling multilingual batches
- Training utilities with the Trainer API
- Evaluation utilities to measure perplexity and token distribution

## Usage Examples

### Basic Tokenization

```python
from src.switch_tokenizer import SwitchableTokenizer

tokenizer = SwitchableTokenizer(
    en_tokenizer_path="gpt2",
    ru_tokenizer_path="DeepPavlov/rubert-base-cased",
)

# Tokenize English text
en_tokens = tokenizer.encode("Hello world", language="EN")
print(f"English: {tokenizer.decode(en_tokens)}")

# Tokenize Russian text
ru_tokens = tokenizer.encode("Привет мир", language="RU")
print(f"Russian: {tokenizer.decode(ru_tokens)}")
```

### Training a Model

```python
from src.model_utils import create_model_with_switchable_tokenizer
from src.data_utils import prepare_multilingual_datasets, SwitchableDataCollator
from transformers import Trainer, TrainingArguments

# Initialize tokenizer and datasets
tokenizer = SwitchableTokenizer(...)
datasets = prepare_multilingual_datasets(tokenizer, ...)

# Create model with the shared vocabulary
model = create_model_with_switchable_tokenizer(
    model_name_or_path="gpt2",
    tokenizer=tokenizer,
)

# Train the model
trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=SwitchableDataCollator(tokenizer),
    train_dataset=datasets["train"],
    eval_dataset=datasets["test"],
)

trainer.train()
```

### Text Generation

```python
from src.model_utils import generate_text

# Generate text in a specific language
en_text = generate_text(
    model=model,
    tokenizer=tokenizer,
    prompt="The weather today is",
    language="EN",
    max_new_tokens=50,
)

# Generate text in another language
ru_text = generate_text(
    model=model,
    tokenizer=tokenizer,
    prompt="Погода сегодня",
    language="RU",
    max_new_tokens=50,
)
```

## Experiments and Evaluation

The implementation includes scripts for evaluating models trained with the switchable tokenizer:

1. **Context-Dependent Probability Distribution Analysis**: Examines how token probabilities shift depending on language context
2. **Perplexity Measurements**: Calculate language-specific and combined perplexity
3. **Language Mixing Evaluation**: Tests the model's ability to handle language switching
4. **ID Overlap Analysis**: Examines how the model handles IDs with different meanings in different languages

## Implementation Challenges

The main challenges in this approach include:

1. **Learning Context-Dependence**: The model needs to learn to maintain language context throughout a sequence
2. **Same ID, Different Meanings**: Each embedding vector represents different tokens in different languages
3. **Tokenization Efficiency**: Without explicit vocabulary merging, repeated tokens across languages don't get automatically shared

## Reference Papers

1. "Towards Making the Most of Multilingual Pretraining for Zero-Shot Neural Machine Translation" (Tang et al., 2020)
2. "The MultiBERTs: BERT Reproductions for Robustness Analysis" (Sellam et al., 2021)
3. "MuLER: Multilingual Early Language Representations" (Srinivasan et al., 2022)

## Further Examples

For detailed examples, check the [examples.md](examples.md) file.

## License

This project is licensed under the MIT License - see the LICENSE file for details. 