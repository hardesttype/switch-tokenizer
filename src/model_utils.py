from typing import Optional, Union, Dict, Any
import torch
from transformers import AutoModelForCausalLM, AutoConfig
from transformers.modeling_utils import PreTrainedModel
from transformers.generation import GenerationConfig
from .switch_tokenizer import SwitchableTokenizer

def create_model_with_switchable_tokenizer(
    model_name_or_path: str,
    tokenizer: SwitchableTokenizer,
    model_config: Optional[Dict[str, Any]] = None,
    device_map: Optional[Union[str, Dict[str, Union[int, str]]]] = None,
    from_scratch: bool = False,
) -> PreTrainedModel:
    """
    Create a language model configured to work with the switchable tokenizer.
    
    Args:
        model_name_or_path: HuggingFace model name or path to model directory
        tokenizer: SwitchableTokenizer instance
        model_config: Optional configuration overrides for the model
        device_map: Optional device map for loading the model
        from_scratch: Whether to initialize the model from scratch instead of using pretrained weights
        
    Returns:
        PreTrainedModel instance configured with the tokenizer's vocabulary size
    """
    if from_scratch:
        # Load config from the model architecture
        config = AutoConfig.from_pretrained(model_name_or_path)
        
        # Initialize model from scratch with the config
        model = AutoModelForCausalLM.from_config(config)
    else:
        # Load the model with its original config and pretrained weights
        model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            device_map=device_map,
        )
    
    # Now resize the token embeddings to match the switchable tokenizer vocab size
    model.resize_token_embeddings(tokenizer.vocab_size)
    
    # Update the model's config to reflect the new vocab size
    model.config.vocab_size = tokenizer.vocab_size
    
    # Apply any additional config overrides if provided
    if model_config:
        for key, value in model_config.items():
            setattr(model.config, key, value)
    
    return model

def setup_generation_config(
    model: PreTrainedModel,
    tokenizer: SwitchableTokenizer,
    language: str,
    generation_config: Optional[Dict[str, Any]] = None,
) -> GenerationConfig:
    """
    Set up generation configuration for language-specific text generation.
    
    Args:
        model: The language model
        tokenizer: SwitchableTokenizer instance
        language: Language code ("EN" or "RU")
        generation_config: Optional generation parameters
        
    Returns:
        GenerationConfig instance
    """
    # Get the language token ID
    lang_token_id = tokenizer.lang_map[language]["lang_token_id"]
    
    # Start with model's default generation config
    config = model.generation_config
    
    # Set the language token as the prefix token
    config.prefix_token_id = lang_token_id
    
    # Apply any additional generation config
    if generation_config:
        for key, value in generation_config.items():
            setattr(config, key, value)
    
    return config

def generate_text(
    model: PreTrainedModel,
    tokenizer: SwitchableTokenizer,
    prompt: str,
    language: str,
    max_new_tokens: int = 100,
    **generation_kwargs
) -> str:
    """
    Generate text with a specific language context.
    
    Args:
        model: The language model
        tokenizer: SwitchableTokenizer instance
        prompt: Text prompt for generation
        language: Language code ("EN" or "RU")
        max_new_tokens: Maximum number of new tokens to generate
        **generation_kwargs: Additional keyword arguments for generation
        
    Returns:
        Generated text as a string
    """
    # Tokenize the prompt with the language context
    input_ids = tokenizer.encode(prompt, language=language)
    input_ids = torch.tensor([input_ids]).to(model.device)
    
    # Set up generation config
    generation_config = setup_generation_config(
        model=model,
        tokenizer=tokenizer,
        language=language,
        generation_config={"max_new_tokens": max_new_tokens, **generation_kwargs},
    )
    
    # Generate text
    output_ids = model.generate(
        input_ids,
        generation_config=generation_config,
    )
    
    # Decode only the newly generated tokens
    generated_ids = output_ids[0, len(input_ids[0]):]
    generated_text = tokenizer.decode(generated_ids.tolist())
    
    return generated_text 