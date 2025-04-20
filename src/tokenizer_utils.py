#!/usr/bin/env python
"""
Utilities for training tokenizers for the experiments.
"""

import os
from typing import List, Dict, Any, Optional
import tempfile
from datasets import load_dataset
from tokenizers import Tokenizer, decoders, models, normalizers, pre_tokenizers, processors, trainers
from transformers import AutoTokenizer, PreTrainedTokenizerFast

def load_multilingual_corpus(
    en_dataset_name: str,
    ru_dataset_name: str,
    en_subset: Optional[str] = None,
    ru_subset: Optional[str] = None,
    limit: int = 10000,
    text_column: str = "text",
    first_shard_only: bool = False,
) -> Dict[str, List[str]]:
    """
    Load English and Russian text datasets for tokenizer training.
    
    Args:
        en_dataset_name: Name of the English dataset on Hugging Face
        ru_dataset_name: Name of the Russian dataset on Hugging Face
        en_subset: Subset of the English dataset, if applicable
        ru_subset: Subset of the Russian dataset, if applicable
        limit: Maximum number of examples to load per language
        text_column: Name of the text column in the datasets
        first_shard_only: If True, load only the first shard (train-00000) instead of counting examples
        
    Returns:
        Dictionary with 'en' and 'ru' keys containing lists of texts
    """
    # Load English dataset with appropriate configuration
    if first_shard_only:
        if en_subset:
            en_dataset = load_dataset(
                en_dataset_name, 
                name=en_subset, 
                data_files=f"{en_subset}/train-00000-of-*.parquet",
                split="train",
                verification_mode="no_checks"
            )
        else:
            en_dataset = load_dataset(
                en_dataset_name, 
                data_files="train-00000-of-*.parquet",
                split="train",
                verification_mode="no_checks"
            )
        
        # Apply the limit even for first_shard_only mode
        if limit and limit < len(en_dataset):
            en_dataset = en_dataset.select(range(limit))
        
        # Load Russian dataset with data_files
        if ru_subset:
            ru_dataset = load_dataset(
                ru_dataset_name, 
                name=ru_subset, 
                data_files=f"{ru_subset}/train-00000-of-*.parquet",
                split="train",
                verification_mode="no_checks"
            )
        else:
            ru_dataset = load_dataset(
                ru_dataset_name, 
                data_files="train-00000-of-*.parquet",
                split="train",
                verification_mode="no_checks"
            )
            
        # Apply the limit even for first_shard_only mode
        if limit and limit < len(ru_dataset):
            ru_dataset = ru_dataset.select(range(limit))
    else:
        # Use regular split with limit for non-shard mode
        en_split = f"train[:{limit}]"
        ru_split = f"train[:{limit}]"
        
        if en_subset:
            en_dataset = load_dataset(en_dataset_name, name=en_subset, split=en_split)
        else:
            en_dataset = load_dataset(en_dataset_name, split=en_split)
        
        # Load Russian dataset
        if ru_subset:
            ru_dataset = load_dataset(ru_dataset_name, name=ru_subset, split=ru_split)
        else:
            ru_dataset = load_dataset(ru_dataset_name, split=ru_split)
    
    # Extract text data
    en_texts = en_dataset[text_column]
    ru_texts = ru_dataset[text_column]
    
    return {
        "en": en_texts,
        "ru": ru_texts,
    }

def train_bpe_tokenizer(
    texts: List[str],
    vocab_size: int = 64000,
    min_frequency: int = 2,
    special_tokens: List[str] = None,
    output_dir: Optional[str] = None,
    file_prefix: str = "tokenizer",
) -> Tokenizer:
    """
    Train a BPE tokenizer on the provided texts.
    
    Args:
        texts: List of texts to train on
        vocab_size: Size of the vocabulary
        min_frequency: Minimum frequency for a token to be included
        special_tokens: List of special tokens to add
        output_dir: Directory to save the tokenizer
        file_prefix: Prefix for the saved tokenizer files
        
    Returns:
        Trained tokenizer
    """
    # Initialize a tokenizer
    tokenizer = Tokenizer(models.BPE())
    
    # Set up normalizers
    tokenizer.normalizer = normalizers.Sequence([
        normalizers.NFD(), 
        normalizers.Lowercase(), 
        normalizers.StripAccents()
    ])
    
    # Set up pre-tokenizers
    tokenizer.pre_tokenizer = pre_tokenizers.Sequence([
        pre_tokenizers.ByteLevel(add_prefix_space=True),
        pre_tokenizers.Digits(individual_digits=True),
    ])
    
    # Configure the trainer
    trainer_instance = trainers.BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=min_frequency,
        special_tokens=special_tokens or ["<unk>", "<pad>", "<s>", "</s>"],
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
    )
    
    # Create a temporary file for training
    with tempfile.NamedTemporaryFile(mode="w+", delete=False, encoding="utf-8") as f:
        for text in texts:
            f.write(text + "\n")
        temp_file = f.name
    
    # Train the tokenizer
    tokenizer.train([temp_file], trainer=trainer_instance)
    
    # Delete the temporary file
    os.unlink(temp_file)
    
    # Set up the decoder
    tokenizer.decoder = decoders.ByteLevel()
    
    # Add post-processor
    tokenizer.post_processor = processors.ByteLevel(trim_offsets=True)
    
    # Save the tokenizer if output_dir is provided
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        tokenizer.save(os.path.join(output_dir, f"{file_prefix}.json"))
    
    return tokenizer

def train_separate_tokenizers(
    corpus: Dict[str, List[str]],
    output_dir: str,
    vocab_size: int = 64000,
    special_tokens: List[str] = None,
) -> Dict[str, str]:
    """
    Train separate tokenizers for each language.
    
    Args:
        corpus: Dictionary with language codes and texts
        output_dir: Directory to save the tokenizers
        vocab_size: Size of the vocabulary for each tokenizer
        special_tokens: List of special tokens to add
        
    Returns:
        Dictionary with paths to the trained tokenizers
    """
    tokenizer_paths = {}
    
    for lang, texts in corpus.items():
        print(f"Training {lang} tokenizer with {len(texts)} texts...")
        
        # Create language-specific output directory
        lang_dir = os.path.join(output_dir, f"{lang}_tokenizer")
        os.makedirs(lang_dir, exist_ok=True)
        
        # Train the tokenizer
        tokenizer = train_bpe_tokenizer(
            texts=texts,
            vocab_size=vocab_size,
            special_tokens=special_tokens,
            output_dir=lang_dir,
            file_prefix=f"{lang}_tokenizer",
        )
        
        # Save the tokenizer and convert to PreTrainedTokenizerFast
        pretrained_tokenizer = PreTrainedTokenizerFast(
            tokenizer_object=tokenizer,
            unk_token="<unk>",
            pad_token="<pad>",
            bos_token="<s>",
            eos_token="</s>",
            additional_special_tokens=special_tokens if special_tokens else [],
        )
        
        # Save the tokenizer in Hugging Face format
        pretrained_tokenizer.save_pretrained(lang_dir)
        
        tokenizer_paths[lang] = lang_dir
    
    return tokenizer_paths

def train_concatenated_tokenizer(
    corpus: Dict[str, List[str]],
    output_dir: str,
    vocab_size: int = 128000,
    special_tokens: List[str] = None,
) -> str:
    """
    Train a tokenizer on concatenated texts from multiple languages.
    
    Args:
        corpus: Dictionary with language codes and texts
        output_dir: Directory to save the tokenizer
        vocab_size: Size of the vocabulary (typically larger for multilingual)
        special_tokens: List of special tokens to add
        
    Returns:
        Path to the trained tokenizer
    """
    # Concatenate all texts
    all_texts = []
    for lang, texts in corpus.items():
        all_texts.extend(texts)
    
    print(f"Training concatenated tokenizer with {len(all_texts)} texts...")
    
    # Create output directory
    concat_dir = os.path.join(output_dir, "concatenated_tokenizer")
    os.makedirs(concat_dir, exist_ok=True)
    
    # Add language tokens to special tokens
    if special_tokens is None:
        special_tokens = []
    
    lang_tokens = [f"<LANG_{lang.upper()}>" for lang in corpus.keys()]
    all_special_tokens = special_tokens + lang_tokens
    
    # Train the tokenizer
    tokenizer = train_bpe_tokenizer(
        texts=all_texts,
        vocab_size=vocab_size,
        special_tokens=["<unk>", "<pad>", "<s>", "</s>"] + all_special_tokens,
        output_dir=concat_dir,
        file_prefix="concat_tokenizer",
    )
    
    # Convert to PreTrainedTokenizerFast
    pretrained_tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        unk_token="<unk>",
        pad_token="<pad>",
        bos_token="<s>",
        eos_token="</s>",
        additional_special_tokens=all_special_tokens,
    )
    
    # Save the tokenizer in Hugging Face format
    pretrained_tokenizer.save_pretrained(concat_dir)
    
    return concat_dir

def create_switchable_tokenizer_from_trained(
    en_tokenizer_path: str,
    ru_tokenizer_path: str,
    output_dir: str,
) -> str:
    """
    Create a SwitchableTokenizer from trained tokenizers.
    
    Args:
        en_tokenizer_path: Path to the English tokenizer
        ru_tokenizer_path: Path to the Russian tokenizer
        output_dir: Directory to save the switchable tokenizer
        
    Returns:
        Path to the switchable tokenizer
    """
    from .switch_tokenizer import SwitchableTokenizer
    
    # Create output directory
    switchable_dir = os.path.join(output_dir, "switchable_tokenizer")
    os.makedirs(switchable_dir, exist_ok=True)
    
    # Create the switchable tokenizer
    tokenizer = SwitchableTokenizer(
        en_tokenizer_path=en_tokenizer_path,
        ru_tokenizer_path=ru_tokenizer_path,
        shared_vocab_size=None,  # Automatically determine the maximum safe shared vocab size
    )
    
    # Save the tokenizer
    tokenizer.save_pretrained(switchable_dir)
    
    return switchable_dir

def analyze_tokenizer_overlap(
    en_tokenizer_path: str,
    ru_tokenizer_path: str,
    sample_texts: Dict[str, List[str]],
    output_file: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Analyze the overlap between two tokenizers.
    
    Args:
        en_tokenizer_path: Path to the English tokenizer
        ru_tokenizer_path: Path to the Russian tokenizer
        sample_texts: Dictionary with sample texts for each language
        output_file: File to save the analysis results
        
    Returns:
        Dictionary with analysis results
    """
    # Load tokenizers
    en_tokenizer = AutoTokenizer.from_pretrained(en_tokenizer_path)
    ru_tokenizer = AutoTokenizer.from_pretrained(ru_tokenizer_path)
    
    # Get vocabulary sets
    en_vocab = set(en_tokenizer.get_vocab().keys())
    ru_vocab = set(ru_tokenizer.get_vocab().keys())
    
    # Calculate overlap
    overlap = en_vocab.intersection(ru_vocab)
    
    # Analyze tokenization efficiency
    en_tokens_per_word = []
    ru_tokens_per_word = []
    
    # Process English texts
    for text in sample_texts["en"]:
        words = text.split()
        if not words:
            continue
            
        tokens = en_tokenizer.tokenize(text)
        en_tokens_per_word.append(len(tokens) / len(words))
    
    # Process Russian texts
    for text in sample_texts["ru"]:
        words = text.split()
        if not words:
            continue
            
        tokens = ru_tokenizer.tokenize(text)
        ru_tokens_per_word.append(len(tokens) / len(words))
    
    # Calculate average tokens per word
    avg_en_tokens_per_word = sum(en_tokens_per_word) / len(en_tokens_per_word) if en_tokens_per_word else 0
    avg_ru_tokens_per_word = sum(ru_tokens_per_word) / len(ru_tokens_per_word) if ru_tokens_per_word else 0
    
    # Compile analysis results
    results = {
        "en_vocab_size": len(en_vocab),
        "ru_vocab_size": len(ru_vocab),
        "overlap_size": len(overlap),
        "overlap_percentage_en": len(overlap) / len(en_vocab) * 100,
        "overlap_percentage_ru": len(overlap) / len(ru_vocab) * 100,
        "avg_en_tokens_per_word": avg_en_tokens_per_word,
        "avg_ru_tokens_per_word": avg_ru_tokens_per_word,
        "sample_overlap_tokens": list(overlap)[:20],  # First 20 overlapping tokens as a sample
    }
    
    # Save analysis results if output_file is provided
    if output_file:
        with open(output_file, "w", encoding="utf-8") as f:
            f.write("=== Tokenizer Overlap Analysis ===\n\n")
            f.write(f"English Vocabulary Size: {results['en_vocab_size']}\n")
            f.write(f"Russian Vocabulary Size: {results['ru_vocab_size']}\n")
            f.write(f"Overlap Size: {results['overlap_size']}\n")
            f.write(f"Overlap Percentage (English): {results['overlap_percentage_en']:.2f}%\n")
            f.write(f"Overlap Percentage (Russian): {results['overlap_percentage_ru']:.2f}%\n")
            f.write(f"Average English Tokens per Word: {results['avg_en_tokens_per_word']:.2f}\n")
            f.write(f"Average Russian Tokens per Word: {results['avg_ru_tokens_per_word']:.2f}\n")
            f.write("\nSample Overlapping Tokens:\n")
            for token in results["sample_overlap_tokens"]:
                f.write(f"  {token}\n")
    
    return results 