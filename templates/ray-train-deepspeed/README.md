# Getting Started with DeepSpeed ZeRO and Ray Train

This template shows how to combine DeepSpeed ZeRO with Ray Train to scale PyTorch training efficiently across GPUs and nodes while minimizing memory usage.

DeepSpeed is a deep learning optimization library focused on scaling and efficiency. Its ZeRO (Zero Redundancy Optimizer) family partitions model states, gradients, and optimizer states across workers to drastically reduce memory usage while maintaining data-parallel semantics. Higher ZeRO stages (e.g., Stage 2/3) remove redundant copies and optionally offload states to CPU, enabling much larger models and batch sizes. DeepSpeed also integrates mixed-precision, communication overlap, and activation checkpointing to improve throughput and lower memory footprint.

This tutorial provides a step-by-step guide on integrating DeepSpeed ZeRO with Ray Train. Specifically, it covers:
- A hands-on example of training an image classification model
- Checkpoint saving and resuming with Ray Train
- Configuring ZeRO for memory and performance (stages, mixed precision, CPU offload)
- Optional GPU memory profiling
- Launching a distributed training job

Note: This template is optimized for the Anyscale platform. When running on open source Ray, you must configure a Ray cluster, install dependencies on all nodes, and set up storage for checkpoints.

**Anyscale Specific Configuration**

Note: This tutorial is optimized for the Anyscale platform. When running on open source Ray, additional configuration is required. For example, you will need to manually:

- **Configure your Ray Cluster**: Set up your multi-node environment and manage resource allocation without Anyscale's automation.
- **Manage Dependencies**: Manually install and manage dependencies on each node.
- **Set Up Storage**: Configure your own distributed or shared storage system for model checkpointing.

## Example Overview

For demonstration purposes, we will integrate Ray Train with DeepSpeed ZeRO using a **Vision Transformer (ViT)** trained on the FashionMNIST dataset. We chose ViT because it has clear, repeatable block structures (transformer blocks) that are ideal for demonstrating ZeRO's partitioning and memory-efficiency capabilities.

While this is a relatively simple example, DeepSpeed configuration (e.g., ZeRO stage, precision, and offloading) can lead to common challenges such as out-of-memory (OOM) errors or suboptimal throughput. Throughout this guide, we'll address these by tuning ZeRO stage (2/3), micro-batch size, mixed precision (bf16/fp16), and optional CPU offloading to balance memory usage and performance for your specific environment.

Install the required dependencies for this tutorial:

```bash
%%bash
pip install torch torchvision matplotlib
pip install transformers datasets==3.6.0 trl
pip install deepspeed
```

### 1. Import packages

In out Python script, let's import necessary packages and set up a logger.

```python
import os
import tempfile
import uuid
import logging

import argparse
from typing import Dict, Any

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
```


### 2. Set up dataloader

The `setup_dataloader` function initializes and prepares a PyTorch `DataLoader` for training.

It starts by fetching the tokenizer for the specified `model_name`. It then loads the `ag_news` dataset, tokenizes the text data, and formats it for PyTorch.
Finally, it wraps the dataset in a `DataLoader` and uses `ray.train.torch.prepare_data_loader` to make it compatible with distributed training in Ray Train.

In this example, we use 

```python
def setup_dataloader(model_name: str, dataset_name: str, seq_length: int, batch_size: int) -> DataLoader:
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    dataset = load_dataset(dataset_name, split="train[:100%]")

    def tokenize_function(examples):
        return tokenizer(examples['text'], padding='max_length', max_length=seq_length, truncation=True)
    
    tokenized_dataset = dataset.map(tokenize_function, batched=True, num_proc=1, keep_in_memory=True)
    tokenized_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask'])

    data_loader = DataLoader(
        tokenized_dataset, 
        batch_size=batch_size,
        shuffle=True
    )

    return ray.train.torch.prepare_data_loader(data_loader)
```


### 3. Model initialization

The `setup_model_and_optimizer` function prepares the model for training. It loads a pretrained causal language model from Hugging Face and initializes an AdamW optimizer. It then uses `deepspeed.initialize` to configure the model and optimizer with the provided DeepSpeed configuration. The function returns the `DeepSpeedEngine`, which manages the model during distributed training.

```python
def setup_model_and_optimizer(model_name: str, learning_rate: float, ds_config: Dict[str, Any]) -> deepspeed.runtime.engine.DeepSpeedEngine:
    model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True)
    log_rank0(f"Model loaded: {model_name} (#parameters: {sum(p.numel() for p in model.parameters())}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    ds_engine, optimizer, _, _ = deepspeed.initialize(
        model=model,
        optimizer=optimizer,
        config=ds_config,
    )
    return ds_engine
```


## 4. Checkpointing and Loading

Checkpointing is crucial for fault tolerance and resuming training. The `report_metrics_and_save_checkpoint` function saves the model's state using `ds_engine.save_checkpoint` into a temporary directory and then reports it to Ray Train along with performance metrics.
 
 ```python
 def report_metrics_and_save_checkpoint(
    ds_engine: deepspeed.runtime.engine.DeepSpeedEngine,
    metrics: Dict[str, Any]
) -> None:
    with tempfile.TemporaryDirectory() as tmp:
        tmp_epoch = os.path.join(tmp, "epoch")
        os.makedirs(tmp_epoch, exist_ok=True)
        ds_engine.save_checkpoint(tmp_epoch)
        torch.distributed.barrier()
        ray.train.report(metrics, checkpoint=Checkpoint.from_directory(tmp))

    log_rank0(f"Checkpoint saved successfully. Metrics: {metrics}")
```

The `load_checkpoint` function handles restoring the training state by loading a Ray Train `Checkpoint` into the DeepSpeed engine, allowing the training to resume from a previously saved state.

```python
def load_checkpoint( ds_engine: deepspeed.runtime.engine.DeepSpeedEngine, ckpt: ray.train.Checkpoint):
    try:
        with ckpt.as_directory() as checkpoint_dir:
            ds_engine.load_checkpoint(checkpoint_dir)

        torch.distributed.barrier()
        log_rank0("Successfully loaded distributed checkpoint")
    except Exception as e:
        logger.error(f"Failed to load checkpoint: {e}")
        raise RuntimeError(f"Checkpoint loading failed: {e}") from e
```


### 5. Training Iteration

The `train_loop` function orchestrates the entire training process. It begins by setting up the dataloader, model, and optimizer. If a checkpoint exists, it restores the training state. The function then iterates through the specified number of epochs, and for each epoch, it loops over the training data.

In each step, it performs a forward pass to compute the loss, followed by a backward pass and an optimizer step to update the model weights. At the end of each epoch, it reports the average loss and saves a checkpoint.

```python
def train_loop(config: Dict[str, Any]) -> None:

    # Load checkpoint if exists
    ckpt = ray.train.get_checkpoint()
    if ckpt:
        load_checkpoint(ds_engine, ckpt)

    train_loader = setup_dataloader(config["model_name"], config["seq_length"], config["batch_size"])
    ds_engine = setup_model_and_optimizer(config["model_name"], config["learning_rate"], config["ds_config"])
    device = ray.train.torch.get_device()

    for epoch in range(config["epochs"]):
        if ray.train.get_context().get_world_size() > 1:
            train_loader.sampler.set_epoch(epoch)

        running_loss = 0.0
        num_batches = 0
        for step, batch in enumerate(train_loader):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            outputs = ds_engine(input_ids=input_ids, attention_mask=attention_mask, labels=input_ids, use_cache=False)
            loss = outputs.loss
            log_rank0(f"step {step} loss: {loss.item()}")

            ds_engine.backward(loss)
            ds_engine.step()

            running_loss += loss.item()
            num_batches += 1

        report_metrics_and_save_checkpoint(ds_engine, {"loss": running_loss / num_batches, "epoch": epoch})
```

## 3. DeepSpeed config

```python
DEEPSPEED_CONFIG = {
    "train_batch_size": "auto",
    "train_micro_batch_size_per_gpu": "auto",
    "bf16": {"enabled": False},
    "fp16": {"enabled": True},
    "zero_optimization": {
        "stage": 2,
        "overlap_comm": True,
        "contiguous_gradients": True,
        "reduce_scatter": True,
        "allgather_partitions": True,
        "reduce_bucket_size": 5e7,
        "stage3_prefetch_bucket_size": 5e7,
        "stage3_param_persistence_threshold": 1e6,
        "offload_param": {"device": "none"},
        "offload_optimizer": {"device": "none"},
    },
    "gradient_clipping": 1.0,
}
```

## 4. Launch trainer

```python
scaling_config = ScalingConfig(num_workers=2, use_gpu=True)

train_loop_config = {
    "epochs": 5,
    "learning_rate": 1e-3,
    "batch_size": 128,
}

run_config = RunConfig(
    storage_path="/mnt/cluster_storage/",
    name=f"deepspeed_mnist_{uuid.uuid4().hex[:8]}",
)

trainer = TorchTrainer(
    train_loop_per_worker=train_loop,
    scaling_config=scaling_config,
    train_loop_config=train_loop_config,
    run_config=run_config,
    deepspeed_config=DEEPSPEED_CONFIG,
)

result = trainer.fit()
print("Training finished", result)
```

## Notes

- Enable `bf16` when supported; otherwise use `fp16`.
- Consider ZeRO Stage 3 for larger models.
- Enable CPU offload in `zero_optimization` when GPU memory is constrained.
- Tune micro-batch size to avoid OOMs.
