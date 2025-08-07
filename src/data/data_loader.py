"""
Modern data loading and preprocessing for LLM training
Includes: Efficient tokenization, streaming datasets, instruction tuning formats
"""

import torch
from torch.utils.data import DataLoader, Dataset, DistributedSampler
from transformers import AutoTokenizer
from datasets import load_dataset, Dataset as HFDataset
import numpy as np
from typing import Dict, Any, List, Optional, Union
import logging


class TextDataset(Dataset):
    """Dataset for text generation tasks"""
    
    def __init__(
        self,
        texts: List[str],
        tokenizer: AutoTokenizer,
        max_length: int = 512,
        add_special_tokens: bool = True,
    ):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.add_special_tokens = add_special_tokens

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        
        # Tokenize text
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt',
            add_special_tokens=self.add_special_tokens,
        )
        
        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
        }


class InstructionDataset(Dataset):
    """Dataset for instruction tuning tasks"""
    
    def __init__(
        self,
        data: List[Dict[str, str]],
        tokenizer: AutoTokenizer,
        max_length: int = 512,
        instruction_template: str = "### Instruction:\n{instruction}\n\n### Response:\n{response}",
    ):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.instruction_template = instruction_template

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        
        # Format instruction and response
        if 'instruction' in item and 'response' in item:
            text = self.instruction_template.format(
                instruction=item['instruction'],
                response=item['response']
            )
        elif 'text' in item:
            text = item['text']
        else:
            raise ValueError("Data must contain either 'instruction'/'response' or 'text' fields")
        
        # Tokenize
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt',
            add_special_tokens=True,
        )
        
        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
        }


class StreamingTextDataset:
    """Streaming dataset for large-scale training"""
    
    def __init__(
        self,
        dataset_name: str,
        tokenizer: AutoTokenizer,
        max_length: int = 512,
        split: str = 'train',
        streaming: bool = True,
    ):
        self.dataset = load_dataset(dataset_name, split=split, streaming=streaming)
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __iter__(self):
        for item in self.dataset:
            text = item.get('text', item.get('content', ''))
            if not text:
                continue
                
            # Tokenize
            encoding = self.tokenizer(
                text,
                truncation=True,
                padding='max_length',
                max_length=self.max_length,
                return_tensors='pt',
                add_special_tokens=True,
            )
            
            yield {
                'input_ids': encoding['input_ids'].squeeze(),
                'attention_mask': encoding['attention_mask'].squeeze(),
            }


def create_dataloader(
    config: Dict[str, Any],
    split: str = 'train',
    batch_size: int = 8,
    world_size: int = 1,
    rank: int = 0,
    num_workers: int = 4,
) -> DataLoader:
    """Create a data loader for training or evaluation"""
    
    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        config.get('tokenizer_name', 'gpt2'),
        trust_remote_code=True
    )
    
    # Add padding token if not present
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load dataset based on configuration
    dataset_name = config.get('dataset_name', 'openwebtext')
    max_length = config.get('sequence_length', 512)
    
    if config.get('streaming', False):
        # Use streaming dataset for large datasets
        dataset = StreamingTextDataset(
            dataset_name=dataset_name,
            tokenizer=tokenizer,
            max_length=max_length,
            split=split,
            streaming=True,
        )
        
        # Create iterable data loader
        return DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=num_workers,
        )
    else:
        # Load full dataset
        if dataset_name == 'openwebtext':
            # Example: OpenWebText dataset
            dataset = load_dataset('openwebtext', split=split)
            texts = [item['text'] for item in dataset]
            
            dataset = TextDataset(
                texts=texts,
                tokenizer=tokenizer,
                max_length=max_length,
            )
        elif dataset_name == 'alpaca':
            # Example: Alpaca instruction dataset
            dataset = load_dataset('tatsu-lab/alpaca', split=split)
            data = [
                {
                    'instruction': item['instruction'],
                    'response': item['output'],
                }
                for item in dataset
            ]
            
            dataset = InstructionDataset(
                data=data,
                tokenizer=tokenizer,
                max_length=max_length,
            )
        else:
            # Generic text dataset
            dataset = load_dataset(dataset_name, split=split)
            texts = [item.get('text', item.get('content', '')) for item in dataset]
            texts = [text for text in texts if text]  # Filter empty texts
            
            dataset = TextDataset(
                texts=texts,
                tokenizer=tokenizer,
                max_length=max_length,
            )
        
        # Create sampler for distributed training
        sampler = None
        if world_size > 1:
            sampler = DistributedSampler(
                dataset,
                num_replicas=world_size,
                rank=rank,
                shuffle=(split == 'train'),
            )
        
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=(split == 'train' and sampler is None),
            sampler=sampler,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=True,
        )


def prepare_instruction_data(
    instructions: List[str],
    responses: List[str],
    tokenizer: AutoTokenizer,
    max_length: int = 512,
) -> Dict[str, torch.Tensor]:
    """Prepare instruction tuning data"""
    
    formatted_texts = []
    for instruction, response in zip(instructions, responses):
        text = f"### Instruction:\n{instruction}\n\n### Response:\n{response}"
        formatted_texts.append(text)
    
    # Tokenize all texts
    encodings = tokenizer(
        formatted_texts,
        truncation=True,
        padding='max_length',
        max_length=max_length,
        return_tensors='pt',
        add_special_tokens=True,
    )
    
    return encodings


def create_conversation_data(
    conversations: List[List[Dict[str, str]]],
    tokenizer: AutoTokenizer,
    max_length: int = 512,
) -> List[Dict[str, torch.Tensor]]:
    """Create data for conversational training"""
    
    formatted_conversations = []
    
    for conversation in conversations:
        text = ""
        for turn in conversation:
            role = turn.get('role', 'user')
            content = turn.get('content', '')
            
            if role == 'user':
                text += f"User: {content}\n"
            elif role == 'assistant':
                text += f"Assistant: {content}\n"
        
        formatted_conversations.append(text)
    
    # Tokenize
    dataset = TextDataset(
        texts=formatted_conversations,
        tokenizer=tokenizer,
        max_length=max_length,
    )
    
    return dataset


# Example usage and data preparation functions
def load_common_datasets():
    """Load and prepare common training datasets"""
    
    datasets = {
        'openwebtext': {
            'name': 'openwebtext',
            'description': 'Large-scale web text corpus',
            'size': '40GB',
            'use_case': 'General language modeling',
        },
        'c4': {
            'name': 'c4',
            'description': 'Common Crawl cleaned dataset',
            'size': '750GB',
            'use_case': 'Large-scale pretraining',
        },
        'alpaca': {
            'name': 'tatsu-lab/alpaca',
            'description': 'Instruction following dataset',
            'size': '52K examples',
            'use_case': 'Instruction tuning',
        },
        'dolly': {
            'name': 'databricks/databricks-dolly-15k',
            'description': 'High-quality instruction dataset',
            'size': '15K examples',
            'use_case': 'Instruction tuning',
        },
        'code': {
            'name': 'codeparrot/github-code',
            'description': 'Code from GitHub repositories',
            'size': '180GB',
            'use_case': 'Code generation',
        },
    }
    
    return datasets


if __name__ == "__main__":
    # Example usage
    from transformers import AutoTokenizer
    
    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token
    
    # Example text dataset
    texts = [
        "This is a sample text for training.",
        "Another example sentence for the model to learn from.",
        "Language models learn patterns from text data.",
    ]
    
    dataset = TextDataset(texts, tokenizer, max_length=128)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True)
    
    # Test data loading
    for batch in dataloader:
        print("Input IDs shape:", batch['input_ids'].shape)
        print("Attention mask shape:", batch['attention_mask'].shape)
        break
