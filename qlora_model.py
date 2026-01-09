"""
QLoRA model for LLM preference classification.

Supports:
- Llama-2/3 (7B, 13B)
- Mistral (7B)
- Gemma (2B, 7B)
- Qwen (1.5B, 7B)
- Phi (2B)

Uses 4-bit quantization + LoRA for efficient finetuning.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import (
    AutoModelForSequenceClassification,
    AutoModelForCausalLM,
    AutoTokenizer,
    AutoConfig,
    BitsAndBytesConfig
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    TaskType
)
from typing import Optional, Tuple, List, Dict
import warnings


# Model configurations for different architectures
MODEL_CONFIGS = {
    # Llama models
    'meta-llama/Llama-2-7b-hf': {
        'target_modules': ['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj'],
        'task_type': TaskType.SEQ_CLS,
        'max_length': 2048,
    },
    'meta-llama/Meta-Llama-3-8B': {
        'target_modules': ['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj'],
        'task_type': TaskType.SEQ_CLS,
        'max_length': 2048,
    },
    # Mistral
    'mistralai/Mistral-7B-v0.1': {
        'target_modules': ['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj'],
        'task_type': TaskType.SEQ_CLS,
        'max_length': 2048,
    },
    # Gemma
    'google/gemma-2b': {
        'target_modules': ['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj'],
        'task_type': TaskType.SEQ_CLS,
        'max_length': 2048,
    },
    'google/gemma-7b': {
        'target_modules': ['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj'],
        'task_type': TaskType.SEQ_CLS,
        'max_length': 2048,
    },
    # Qwen
    'Qwen/Qwen1.5-1.8B': {
        'target_modules': ['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj'],
        'task_type': TaskType.SEQ_CLS,
        'max_length': 2048,
    },
    'Qwen/Qwen1.5-7B': {
        'target_modules': ['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj'],
        'task_type': TaskType.SEQ_CLS,
        'max_length': 2048,
    },
    # Phi
    'microsoft/phi-2': {
        'target_modules': ['q_proj', 'k_proj', 'v_proj', 'dense', 'fc1', 'fc2'],
        'task_type': TaskType.SEQ_CLS,
        'max_length': 2048,
    },
}

# Default config for unknown models
DEFAULT_CONFIG = {
    'target_modules': ['q_proj', 'k_proj', 'v_proj', 'o_proj'],
    'task_type': TaskType.SEQ_CLS,
    'max_length': 2048,
}


def get_model_config(model_name: str) -> dict:
    """Get model-specific configuration."""
    for key in MODEL_CONFIGS:
        if key in model_name or model_name in key:
            return MODEL_CONFIGS[key]
    return DEFAULT_CONFIG


class QLoRAClassifier(nn.Module):
    """
    QLoRA-based classifier for LLM preference prediction.

    Uses a pretrained LLM with 4-bit quantization and LoRA adapters.
    """

    def __init__(
        self,
        model_name: str,
        num_labels: int = 3,
        lora_r: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.1,
        use_4bit: bool = True,
        use_nested_quant: bool = True,
        bnb_4bit_compute_dtype: str = "float16",
        gradient_checkpointing: bool = True,
        use_flash_attention: bool = False,
    ):
        super().__init__()

        self.model_name = model_name
        self.num_labels = num_labels
        model_config = get_model_config(model_name)

        # BitsAndBytes quantization config
        compute_dtype = getattr(torch, bnb_4bit_compute_dtype)

        bnb_config = None
        if use_4bit:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=compute_dtype,
                bnb_4bit_use_double_quant=use_nested_quant,
            )

        # Load model config
        config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
        config.num_labels = num_labels
        config.pad_token_id = config.eos_token_id

        # Load model
        attn_implementation = "flash_attention_2" if use_flash_attention else "eager"

        try:
            # Try loading as sequence classification model
            self.model = AutoModelForSequenceClassification.from_pretrained(
                model_name,
                config=config,
                quantization_config=bnb_config,
                device_map="auto",
                trust_remote_code=True,
                attn_implementation=attn_implementation,
            )
        except Exception:
            # Fall back to causal LM and add classification head
            warnings.warn(f"Could not load {model_name} as SeqCls, using CausalLM with custom head")
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                quantization_config=bnb_config,
                device_map="auto",
                trust_remote_code=True,
                attn_implementation=attn_implementation,
            )
            # Add classification head
            hidden_size = self.model.config.hidden_size
            self.score = nn.Linear(hidden_size, num_labels)
            self._use_custom_head = True
        else:
            self._use_custom_head = False

        # Prepare for k-bit training
        if use_4bit:
            self.model = prepare_model_for_kbit_training(
                self.model,
                use_gradient_checkpointing=gradient_checkpointing
            )

        # LoRA configuration
        lora_config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            target_modules=model_config['target_modules'],
            lora_dropout=lora_dropout,
            bias="none",
            task_type=model_config['task_type'],
        )

        # Apply LoRA
        self.model = get_peft_model(self.model, lora_config)
        self.model.print_trainable_parameters()

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass.

        Args:
            input_ids: (batch, seq_len)
            attention_mask: (batch, seq_len)
            labels: (batch, num_labels) soft labels or (batch,) class indices

        Returns:
            logits: (batch, num_labels)
            loss: scalar if labels provided
        """
        if self._use_custom_head:
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                return_dict=True
            )
            # Get last token hidden state (for decoder models)
            hidden_states = outputs.hidden_states[-1]

            # Find last non-padding token
            batch_size = input_ids.shape[0]
            sequence_lengths = attention_mask.sum(dim=1) - 1
            pooled = hidden_states[range(batch_size), sequence_lengths]

            logits = self.score(pooled)
        else:
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                return_dict=True
            )
            logits = outputs.logits

        loss = None
        if labels is not None:
            if labels.dim() == 2:
                # Soft labels
                log_probs = F.log_softmax(logits, dim=-1)
                loss = -torch.sum(labels * log_probs, dim=-1).mean()
            else:
                loss = F.cross_entropy(logits, labels)

        return logits, loss

    def predict_proba(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Get class probabilities."""
        self.eval()
        with torch.no_grad():
            logits, _ = self.forward(input_ids, attention_mask)
            probs = F.softmax(logits, dim=-1)
        return probs

    def save_pretrained(self, path: str):
        """Save LoRA adapters."""
        self.model.save_pretrained(path)

    @classmethod
    def from_pretrained(cls, path: str, **kwargs):
        """Load from saved LoRA adapters."""
        from peft import PeftModel

        # This is a simplified loader - full implementation would
        # need to save/load the config as well
        raise NotImplementedError("Use load_qlora_model() function instead")


def get_qlora_model_and_tokenizer(
    model_name: str,
    num_labels: int = 3,
    lora_r: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.1,
    use_4bit: bool = True,
    **kwargs
) -> Tuple[QLoRAClassifier, AutoTokenizer]:
    """
    Initialize QLoRA model and tokenizer.

    Args:
        model_name: HuggingFace model name or path
        num_labels: Number of classification labels
        lora_r: LoRA rank
        lora_alpha: LoRA alpha scaling factor
        lora_dropout: Dropout for LoRA layers
        use_4bit: Whether to use 4-bit quantization

    Returns:
        model: QLoRAClassifier
        tokenizer: AutoTokenizer
    """
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
        padding_side='left'  # Important for decoder models
    )

    # Set pad token if not present
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    model = QLoRAClassifier(
        model_name=model_name,
        num_labels=num_labels,
        lora_r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        use_4bit=use_4bit,
        **kwargs
    )

    return model, tokenizer


class QLoRADataset(torch.utils.data.Dataset):
    """
    Dataset for QLoRA models.

    Uses a different input format optimized for decoder models:
    [INST] Compare these two responses to the prompt.

    Prompt: {prompt}

    Response A: {response_a}

    Response B: {response_b}

    Which response is better? [/INST]
    """

    def __init__(
        self,
        df,
        tokenizer,
        max_length: int = 2048,
        is_test: bool = False,
        input_format: str = "instruct"  # instruct, simple, chat
    ):
        self.df = df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.is_test = is_test
        self.input_format = input_format

        self.label_cols = ['winner_model_a', 'winner_model_b', 'winner_tie']

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        text = self._format_input(
            prompt=str(row['prompt']),
            response_a=str(row['response_a']),
            response_b=str(row['response_b'])
        )

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

        if not self.is_test:
            import numpy as np
            labels = row[self.label_cols].values.astype(np.float32)
            item['labels'] = torch.tensor(labels)

        item['id'] = row['id']

        return item

    def _format_input(self, prompt: str, response_a: str, response_b: str) -> str:
        """Format input based on chosen format."""
        prompt = self._truncate_text(prompt, 1000)
        response_a = self._truncate_text(response_a, 2000)
        response_b = self._truncate_text(response_b, 2000)

        if self.input_format == "instruct":
            return f"""[INST] Compare these two AI assistant responses and determine which one is better.

### User Prompt:
{prompt}

### Response A:
{response_a}

### Response B:
{response_b}

Based on helpfulness, accuracy, and quality, which response is better? Answer with A, B, or Tie. [/INST]"""

        elif self.input_format == "chat":
            return f"""<|system|>You are evaluating AI assistant responses.</s>
<|user|>Compare these responses:

Prompt: {prompt}

Response A: {response_a}

Response B: {response_b}

Which is better?</s>
<|assistant|>"""

        else:  # simple
            return f"""Prompt: {prompt}

Response A: {response_a}

Response B: {response_b}

Comparison:"""

    @staticmethod
    def _truncate_text(text: str, max_chars: int) -> str:
        """Truncate text to max characters."""
        if len(text) > max_chars:
            return text[:max_chars] + "..."
        return text


def get_qlora_dataloaders(
    train_df,
    val_df,
    tokenizer,
    batch_size: int = 2,
    max_length: int = 2048,
    num_workers: int = 4,
    input_format: str = "instruct"
):
    """Create dataloaders for QLoRA training."""
    from torch.utils.data import DataLoader

    train_dataset = QLoRADataset(
        df=train_df,
        tokenizer=tokenizer,
        max_length=max_length,
        is_test=False,
        input_format=input_format
    )

    val_dataset = QLoRADataset(
        df=val_df,
        tokenizer=tokenizer,
        max_length=max_length,
        is_test=False,
        input_format=input_format
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
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    return train_loader, val_loader
