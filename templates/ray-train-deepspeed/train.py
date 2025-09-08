import os
import tempfile
import uuid
import logging

import argparse
from typing import Dict, Any

os.environ["RAY_TRAIN_V2_ENABLED"] = "1"

import ray
import ray.train
import ray.train.torch
from ray.train.torch import TorchTrainer
from ray.train import ScalingConfig, RunConfig, Checkpoint

import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset, DownloadConfig

import deepspeed


logger = logging.getLogger(__name__)


def get_tokenizer(model_name: str, trust_remote_code: bool = True) -> Any:
    """
    Load and configure the tokenizer for the given model.
    
    Args:
        model_name: Name of the model to load tokenizer for
        trust_remote_code: Whether to trust remote code
        
    Returns:
        Configured tokenizer
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=trust_remote_code)
    
    # Set pad token if not already set
    if tokenizer.pad_token is None:
        if tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
        else:
            # Fallback for models without eos_token
            tokenizer.pad_token = tokenizer.convert_ids_to_tokens(2)
    
    return tokenizer


def train_loop(config: Dict[str, Any]) -> None:

    print(f"train_loop config: {config}")

    # TODO: Load checkpoint if exists

    tokenizer = get_tokenizer(config["model_name"], trust_remote_code=True)

    split_str = f"train[:100%]"
    dataset = load_dataset('ag_news', split=split_str, download_config=DownloadConfig(disable_tqdm=True))
    text_column = 'text'

    def tokenize_function(examples):
        return tokenizer(examples[text_column], padding='max_length', max_length=config["seq_length"], truncation=True)
    
    tokenized_dataset = dataset.map(tokenize_function, batched=True, num_proc=1, keep_in_memory=True)
    tokenized_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask'])

    data_loader = DataLoader(
        tokenized_dataset, 
        batch_size=config["batch_size"], 
        shuffle=True
    )
    train_loader = ray.train.torch.prepare_data_loader(data_loader)

    model = AutoModelForCausalLM.from_pretrained(config["model_name"], trust_remote_code=True)

    print(f"#parameters: {sum(p.numel() for p in model.parameters())}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"])
    ds_engine, optimizer, _, _ = deepspeed.initialize(
        model=model,
        optimizer=optimizer,
        config=config["ds_config"],
    )

    device = ray.train.torch.get_device()

    for epoch in range(config["epochs"]):

        sum_loss = 0.0
        num_steps = 0
        for step, batch in enumerate(train_loader):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            outputs = ds_engine(input_ids=input_ids, attention_mask=attention_mask, labels=input_ids, use_cache=False)
            loss = outputs.loss
            print(f"step {step} loss: {loss}")
            ds_engine.backward(loss)
            ds_engine.step()

            sum_loss += loss.item()
            num_steps += 1

        with tempfile.TemporaryDirectory() as tmp:
            tmp_epoch = os.path.join(tmp, "epoch")
            os.makedirs(tmp_epoch, exist_ok=True)
            model.save_checkpoint(tmp_epoch)
            ray.train.report({"loss": sum_loss / num_steps, "epoch": epoch}, checkpoint=Checkpoint.from_directory(tmp))


def main():
    args = get_args()
    print(args)

    scaling_config = ScalingConfig(num_workers=2, use_gpu=True)

    ds_config = {
        "train_micro_batch_size_per_gpu": args.batch_size,
        "bf16": {"enabled": True},
        "grad_accum_dtype": "bf16",
        "zero_optimization": {
            "stage": args.zero_stage,
            "overlap_comm": True,
            "contiguous_gradients": True,
        },
        "gradient_clipping": 1.0,
    }

    train_loop_config = {
        "epochs": args.num_epochs,
        "learning_rate": args.learning_rate,
        "batch_size": args.batch_size,
        "ds_config": ds_config,
        "model_name": args.model_name,
        "seq_length": args.seq_length,
    }

    run_config = RunConfig(
        storage_path="/mnt/cluster_storage/",
        name=f"deepspeed_sample_{uuid.uuid4().hex[:8]}",
    )

    trainer = TorchTrainer(
        train_loop_per_worker=train_loop,
        scaling_config=scaling_config,
        train_loop_config=train_loop_config,
        run_config=run_config,
    )

    result = trainer.fit()
    print("Training finished", result)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="MiniLLM/MiniPLM-Qwen-500M")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_epochs", type=int, default=1)
    parser.add_argument("--seq_length", type=int, default=512)
    parser.add_argument("--learning_rate", type=float, default=1e-6)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--activation_checkpointing", action="store_true")
    parser.add_argument("--eval", action="store_true")
    parser.add_argument("--dataset_name", type=str, default="wikitext", help="Dataset name for pretraining evaluation")
    parser.add_argument("--dataset_percentage", type=float, default=10.0, help="Percentage of dataset to use (e.g., 10.0 for 10 percent)")
    parser.add_argument("--num_layers", type=int, default=0)
    parser.add_argument("--attn_impl", type=str, default="sdpa")
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--passes", type=str, default=None)
    parser.add_argument("--backend", type=str, default="inductor")
    parser.add_argument("--offload_opt_states", action="store_true")
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--profile_dir", type=str, default=None)
    parser.add_argument("--bench_step", type=int, default=100)
    parser.add_argument("--warmup_step", type=int, default=15)
    parser.add_argument("--zero_stage", type=int, default=3)
    parser.add_argument("--log_interval", type=int, default=10)
    parser.add_argument("--save_weights", action="store_true")
    parser.add_argument("--load_weights", action="store_true")
        # WandB logging arguments
    parser.add_argument("--use_wandb", action="store_true", help="Enable wandb logging")
    parser.add_argument("--wandb_project", type=str, default="ds-verify-loss", help="WandB project name")
    parser.add_argument("--wandb_run_name", type=str, default=None, help="WandB run name")
    parser.add_argument("--wandb_tags", type=str, nargs="+", default=[], help="WandB tags for the run")

    return parser.parse_args()


if __name__ == "__main__":
    main()
