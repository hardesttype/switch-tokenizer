from typing import Dict, List, Optional, Union, Any, Tuple
import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset, Dataset as HFDataset
from transformers import DataCollatorForLanguageModeling

from .switch_tokenizer import SwitchableTokenizer

class MultilingualDataset(Dataset):
    """
    Dataset for training a model with a switchable tokenizer on multilingual data.
    """
    
    def __init__(
        self,
        tokenizer: SwitchableTokenizer,
        datasets: Dict[str, Union[HFDataset, List[str]]],
        max_length: int = 1024,
        text_column: str = "text",
    ):
        """
        Initialize the multilingual dataset.
        
        Args:
            tokenizer: SwitchableTokenizer instance
            datasets: Dictionary mapping language codes to datasets or lists of texts
            max_length: Maximum sequence length for tokenization
            text_column: Name of the text column in the datasets
        """
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.text_column = text_column
        
        # Process and combine datasets
        self.examples = []
        self.language_indices = {}
        
        current_idx = 0
        for lang, dataset in datasets.items():
            if lang not in tokenizer.lang_map:
                raise ValueError(f"Unsupported language: {lang}. Supported: {list(tokenizer.lang_map.keys())}")
            
            # Store start index for this language
            start_idx = current_idx
            
            # Convert the dataset to the internal format
            if isinstance(dataset, HFDataset):
                texts = dataset[text_column]
            elif isinstance(dataset, list):
                texts = dataset
            else:
                raise ValueError(f"Unsupported dataset type for language {lang}: {type(dataset)}")
            
            # Add examples with language context
            for text in texts:
                self.examples.append({
                    "text": text,
                    "language": lang,
                })
                current_idx += 1
            
            # Store the range of indices for this language
            self.language_indices[lang] = (start_idx, current_idx - 1)
    
    def __len__(self) -> int:
        return len(self.examples)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        example = self.examples[idx]
        text = example["text"]
        language = example["language"]
        
        # Tokenize with the appropriate language context
        encoding = self.tokenizer.encode(
            text,
            language=language,
            truncation=True,
            max_length=self.max_length,
        )
        
        return {
            "input_ids": torch.tensor(encoding, dtype=torch.long),
            "language": language,
        }
    
    def get_language_split(self) -> Dict[str, Tuple[int, int]]:
        """
        Get the index ranges for each language in the dataset.
        
        Returns:
            Dictionary mapping language codes to (start_idx, end_idx) tuples
        """
        return self.language_indices

class SwitchableDataCollator(DataCollatorForLanguageModeling):
    """
    Data collator for language modeling with the switchable tokenizer.
    Handles multiple languages in the same batch.
    """
    
    def __init__(
        self,
        tokenizer: SwitchableTokenizer,
        mlm: bool = False,
        mlm_probability: float = 0.15,
        pad_to_multiple_of: Optional[int] = None,
    ):
        """
        Initialize the data collator.
        
        Args:
            tokenizer: SwitchableTokenizer instance
            mlm: Whether to use masked language modeling (vs. causal LM)
            mlm_probability: Probability of masking tokens in MLM
            pad_to_multiple_of: Optional padding multiple
        """
        super().__init__(
            tokenizer=tokenizer,  # Use the switchable tokenizer directly
            mlm=mlm,
            mlm_probability=mlm_probability,
            pad_to_multiple_of=pad_to_multiple_of,
        )
        self.switchable_tokenizer = tokenizer
    
    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        """
        Collate features into a batch.
        
        Args:
            features: List of feature dictionaries, each with "input_ids" and "language"
            
        Returns:
            Batch dictionary with tensors suitable for model input
        """
        # Extract input_ids
        input_ids = [feature["input_ids"] for feature in features]
        
        # Convert to tensors if not already
        input_ids = [ids if isinstance(ids, torch.Tensor) else torch.tensor(ids) for ids in input_ids]
        
        # Create attention masks
        attention_mask = [torch.ones_like(ids) for ids in input_ids]
        
        # Determine max length for padding
        max_length = max(ids.size(0) for ids in input_ids)
        
        # Pad to max length
        input_ids_padded = []
        attention_mask_padded = []
        
        for ids, mask in zip(input_ids, attention_mask):
            pad_length = max_length - ids.size(0)
            if pad_length > 0:
                # Pad with 0 (assuming 0 is the pad token ID)
                input_ids_padded.append(torch.cat([ids, torch.zeros(pad_length, dtype=torch.long)]))
                attention_mask_padded.append(torch.cat([mask, torch.zeros(pad_length, dtype=torch.long)]))
            else:
                input_ids_padded.append(ids)
                attention_mask_padded.append(mask)
        
        # Stack tensors
        batch = {
            "input_ids": torch.stack(input_ids_padded),
            "attention_mask": torch.stack(attention_mask_padded),
        }
        
        # For causal LM, use input_ids as labels
        if not self.mlm:
            batch["labels"] = batch["input_ids"].clone()
            
            # Mask padding in labels (-100 is the ignore index for CrossEntropyLoss)
            for i, mask in enumerate(batch["attention_mask"]):
                batch["labels"][i][mask == 0] = -100
        else:
            # For MLM, create masked input_ids and corresponding labels
            batch["labels"] = batch["input_ids"].clone()
            probability_matrix = torch.full_like(batch["input_ids"], self.mlm_probability, dtype=torch.float)
            special_tokens_mask = self._get_special_tokens_mask(batch["input_ids"])
            probability_matrix.masked_fill_(special_tokens_mask, value=0.0)
            masked_indices = torch.bernoulli(probability_matrix).bool()
            batch["labels"][~masked_indices] = -100
            
            # Replace some masked input tokens with random tokens
            random_words = torch.randint(
                low=0, high=self.switchable_tokenizer.shared_vocab_size, 
                size=(masked_indices.sum(),), dtype=torch.long
            )
            batch["input_ids"][masked_indices] = random_words
        
        return batch
    
    def _get_special_tokens_mask(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Create a mask for special tokens that shouldn't be masked in MLM.
        
        Args:
            input_ids: Input token IDs
            
        Returns:
            Boolean mask with True for special tokens
        """
        special_tokens_mask = torch.zeros_like(input_ids, dtype=torch.bool)
        
        # Mark special tokens with True in the mask
        for special_id in self.switchable_tokenizer.all_special_ids:
            special_tokens_mask |= (input_ids == special_id)
        
        # Also mark padding tokens
        special_tokens_mask |= (input_ids == 0)  # Assume 0 is the pad token ID
        
        return special_tokens_mask

def prepare_multilingual_datasets(
    tokenizer: SwitchableTokenizer,
    dataset_configs: Dict[str, Dict[str, Any]],
    text_column: str = "text",
    max_length: int = 1024,
    train_test_split: Optional[float] = 0.1,
    seed: int = 42,
) -> Dict[str, MultilingualDataset]:
    """
    Prepare datasets for multilingual training and evaluation.
    
    Args:
        tokenizer: SwitchableTokenizer instance
        dataset_configs: Dictionary mapping language codes to dataset configurations
        text_column: Name of the text column in the datasets
        max_length: Maximum sequence length for tokenization
        train_test_split: If provided, fraction of data to use for testing
        seed: Random seed for reproducibility
        
    Returns:
        Dictionary with "train" and optionally "test" MultilingualDataset instances
    """
    train_datasets = {}
    test_datasets = {}
    
    # Check if we're dealing with first_shard_only data
    using_first_shard = any(
        "data_files" in config and ("train-00000" in config["data_files"] or 
        (config.get("name", "") and "train-00000" in config.get("data_files", "")))
        for config in dataset_configs.values()
    )
    
    for lang, config in dataset_configs.items():
        # Load the dataset
        dataset = None
        
        if "path" in config:
            # Check if we have data_files specified
            if "data_files" in config:
                # If we have data_files with no path info and name is provided, add the name to the path
                data_files = config["data_files"]
                if "/" not in data_files and config.get("name"):
                    data_files = f"{config['name']}/{data_files}"
                
                # When using first shard, don't use caching to avoid mismatch errors
                dataset = load_dataset(
                    config["path"],
                    name=config.get("name"),
                    data_files=data_files,
                    split="train",
                    verification_mode="no_checks" if using_first_shard else None
                )
                
                # Apply data limit after loading if specified
                if config.get("apply_limit") and "limit" in config and config["limit"] < len(dataset):
                    dataset = dataset.select(range(config["limit"]))
                
            else:
                # Load from Hugging Face Datasets
                dataset = load_dataset(
                    config["path"],
                    name=config.get("name"),
                    split=config.get("split", "train"),
                )
        elif "texts" in config:
            # Use provided list of texts
            dataset = config["texts"]
        else:
            raise ValueError(f"Invalid dataset config for language {lang}: {config}")
        
        # Split into train and test if requested
        if train_test_split is not None and isinstance(dataset, HFDataset):
            # Make a simple manual split if using first shard to avoid caching issues
            if using_first_shard:
                total_examples = len(dataset)
                test_size = int(total_examples * train_test_split)
                train_size = total_examples - test_size
                
                # Use simple slicing for train/test split
                indices = list(range(total_examples))
                train_indices = indices[:train_size]
                test_indices = indices[train_size:]
                
                train_datasets[lang] = dataset.select(train_indices)
                test_datasets[lang] = dataset.select(test_indices)
            else:
                # Use standard train_test_split for non-shard data
                splits = dataset.train_test_split(test_size=train_test_split, seed=seed)
                train_datasets[lang] = splits["train"]
                test_datasets[lang] = splits["test"]
        else:
            train_datasets[lang] = dataset
            # No test split in this case
    
    # Create MultilingualDataset instances
    result = {
        "train": MultilingualDataset(
            tokenizer=tokenizer,
            datasets=train_datasets,
            max_length=max_length,
            text_column=text_column,
        )
    }
    
    if test_datasets:
        result["test"] = MultilingualDataset(
            tokenizer=tokenizer,
            datasets=test_datasets,
            max_length=max_length,
            text_column=text_column,
        )
    
    return result

def create_data_loaders(
    datasets: Dict[str, MultilingualDataset],
    collator: SwitchableDataCollator,
    batch_size: int = 8,
    num_workers: int = 4,
    shuffle_train: bool = True,
) -> Dict[str, DataLoader]:
    """
    Create data loaders for training and evaluation.
    
    Args:
        datasets: Dictionary with dataset splits
        collator: Data collator instance
        batch_size: Batch size for training
        num_workers: Number of worker processes for data loading
        shuffle_train: Whether to shuffle the training data
        
    Returns:
        Dictionary with DataLoader instances for each split
    """
    loaders = {}
    
    for split, dataset in datasets.items():
        shuffle = shuffle_train and split == "train"
        
        loaders[split] = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            collate_fn=collator,
            pin_memory=True,
        )
    
    return loaders

class FormattedDataset(Dataset):
    """
    A dataset for converting already tokenized or raw text datasets to a format
    compatible with the DataLoader and language models.
    """
    def __init__(self, original_dataset, tokenizer, max_length):
        self.tokenizer = tokenizer
        self.examples = []
        
        # Process each example to ensure proper format
        for i in range(len(original_dataset)):
            try:
                example = original_dataset[i]
                
                # If example already has input_ids, use them directly
                if 'input_ids' in example and isinstance(example['input_ids'], list):
                    # Ensure they're integers, not nested lists or strings
                    input_ids = [int(id) if isinstance(id, (int, float, str)) else id[0] if isinstance(id, list) else 0 
                                for id in example['input_ids']]
                    
                    # Create a proper example
                    self.examples.append({
                        'input_ids': input_ids[:max_length],
                        'attention_mask': [1] * min(len(input_ids), max_length)
                    })
                # If example has text, tokenize it
                elif 'text' in example:
                    encoded = tokenizer(
                        example['text'],
                        truncation=True,
                        max_length=max_length,
                        padding='max_length',
                        return_tensors=None  # Return Python lists
                    )
                    self.examples.append(encoded)
                # Skip examples that don't have the right format
                else:
                    print(f"Skipping example with keys: {example.keys()}")
            except Exception as e:
                print(f"Error processing example: {e}")
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        return self.examples[idx] 