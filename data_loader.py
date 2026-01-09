"""
Data loading utilities for LLM Classification Finetuning.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Tuple, List
from sklearn.model_selection import StratifiedKFold
import torch
from torch.utils.data import Dataset, DataLoader


class PreferenceDataset(Dataset):
    """
    Dataset for LLM preference classification.

    Concatenates prompt + response_a + response_b for classification.

    Supports multiple input formats:
    - default: [PROMPT] prompt [RESPONSE A] response_a [RESPONSE B] response_b
    - markdown: ### Prompt\n{prompt}\n\n### Response A\n...
    - comparison: Compare these responses...
    - simple: prompt [SEP] response_a [SEP] response_b
    """

    def __init__(
        self,
        df: pd.DataFrame,
        tokenizer,
        max_length: int = 1024,
        is_test: bool = False,
        format_type: str = "default"
    ):
        self.df = df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.is_test = is_test
        self.format_type = format_type

        # Target columns
        self.label_cols = ['winner_model_a', 'winner_model_b', 'winner_tie']
        
    def __len__(self) -> int:
        return len(self.df)
    
    def __getitem__(self, idx: int) -> dict:
        row = self.df.iloc[idx]
        
        # Format input text
        text = self._format_input(
            prompt=str(row['prompt']),
            response_a=str(row['response_a']),
            response_b=str(row['response_b'])
        )
        
        # Tokenize
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        item = {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
        }
        
        # Add token_type_ids if available
        if 'token_type_ids' in encoding:
            item['token_type_ids'] = encoding['token_type_ids'].squeeze(0)
        
        # Add labels for training
        if not self.is_test:
            labels = row[self.label_cols].values.astype(np.float32)
            item['labels'] = torch.tensor(labels)
            # Also add class index for stratification
            item['label_idx'] = torch.tensor(np.argmax(labels))
        
        # Keep track of id for submission
        item['id'] = row['id']
        
        return item
    
    def _format_input(
        self,
        prompt: str,
        response_a: str,
        response_b: str
    ) -> str:
        """
        Format the input for the model.

        Multiple formats available for experimentation:
        - default: [PROMPT] prompt [RESPONSE A] response_a [RESPONSE B] response_b
        - markdown: ### Prompt\n{prompt}\n\n### Response A\n...
        - comparison: Compare: {prompt}\n\nOption A: ...
        - simple: {prompt} [SEP] {response_a} [SEP] {response_b}
        """
        # Clean and truncate to prevent tokenizer overflow
        prompt = self._clean_text(prompt)[:2000]
        response_a = self._clean_text(response_a)[:3000]
        response_b = self._clean_text(response_b)[:3000]

        if self.format_type == "markdown":
            text = (
                f"### Prompt\n{prompt}\n\n"
                f"### Response A\n{response_a}\n\n"
                f"### Response B\n{response_b}"
            )
        elif self.format_type == "comparison":
            text = (
                f"Compare these responses to the following prompt:\n\n"
                f"Prompt: {prompt}\n\n"
                f"Response A: {response_a}\n\n"
                f"Response B: {response_b}"
            )
        elif self.format_type == "simple":
            text = f"{prompt} [SEP] {response_a} [SEP] {response_b}"
        else:  # default
            text = (
                f"[PROMPT] {prompt} "
                f"[RESPONSE A] {response_a} "
                f"[RESPONSE B] {response_b}"
            )

        return text
    
    @staticmethod
    def _clean_text(text: str) -> str:
        """Basic text cleaning."""
        if pd.isna(text):
            return ""
        text = str(text)
        # Remove excessive whitespace
        text = ' '.join(text.split())
        return text


def load_data(
    train_path: str,
    test_path: Optional[str] = None
) -> Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
    """
    Load train and test dataframes.
    """
    train_df = pd.read_csv(train_path)
    
    # Create binary target columns if not present
    if 'winner_model_a' not in train_df.columns:
        # Assuming there's a 'winner' column
        if 'winner' in train_df.columns:
            train_df['winner_model_a'] = (train_df['winner'] == 'model_a').astype(int)
            train_df['winner_model_b'] = (train_df['winner'] == 'model_b').astype(int)
            train_df['winner_tie'] = (train_df['winner'] == 'tie').astype(int)
    
    test_df = None
    if test_path and Path(test_path).exists():
        test_df = pd.read_csv(test_path)
    
    return train_df, test_df


def create_folds(
    df: pd.DataFrame,
    n_folds: int = 5,
    seed: int = 42,
    stratified: bool = True
) -> pd.DataFrame:
    """
    Add fold column to dataframe for cross-validation.
    """
    df = df.copy()
    df['fold'] = -1
    
    # Get stratification target
    if stratified:
        # Create single label for stratification
        label_cols = ['winner_model_a', 'winner_model_b', 'winner_tie']
        y = df[label_cols].values.argmax(axis=1)
    else:
        y = np.zeros(len(df))
    
    kfold = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    
    for fold, (_, val_idx) in enumerate(kfold.split(df, y)):
        df.loc[val_idx, 'fold'] = fold
    
    return df


def get_dataloaders(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    tokenizer,
    batch_size: int = 8,
    max_length: int = 1024,
    num_workers: int = 4
) -> Tuple[DataLoader, DataLoader]:
    """
    Create train and validation dataloaders.
    """
    train_dataset = PreferenceDataset(
        df=train_df,
        tokenizer=tokenizer,
        max_length=max_length,
        is_test=False
    )
    
    val_dataset = PreferenceDataset(
        df=val_df,
        tokenizer=tokenizer,
        max_length=max_length,
        is_test=False
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size * 2,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader


def get_test_dataloader(
    test_df: pd.DataFrame,
    tokenizer,
    batch_size: int = 16,
    max_length: int = 1024,
    num_workers: int = 4
) -> DataLoader:
    """
    Create test dataloader for inference.
    """
    test_dataset = PreferenceDataset(
        df=test_df,
        tokenizer=tokenizer,
        max_length=max_length,
        is_test=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return test_loader
