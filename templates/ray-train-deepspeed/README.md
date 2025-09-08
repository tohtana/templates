# Getting Started with DeepSpeed ZeRO and Ray Train

This template shows how to combine DeepSpeed ZeRO with Ray Train to scale PyTorch training efficiently across GPUs and nodes while minimizing memory usage.

DeepSpeed is a deep learning optimization library focused on scaling and efficiency. Its ZeRO (Zero Redundancy Optimizer) family partitions model states, gradients, and optimizer states across workers to drastically reduce memory usage while maintaining data-parallel semantics.

This tutorial provides a step-by-step guide on integrating DeepSpeed ZeRO with Ray Train. Specifically, it covers:
- A hands-on example of fine-tuning a LLM
- Checkpoint saving and resuming with Ray Train
- Configuring ZeRO for memory and performance (stages, mixed precision, CPU offload)
- Launching a distributed training job

Note: This template is optimized for the Anyscale platform. When running on open source Ray, you must configure a Ray cluster, install dependencies on all nodes, and set up storage for checkpoints.

**Anyscale Specific Configuration**

Note: This tutorial is optimized for the Anyscale platform. When running on open source Ray, additional configuration is required. For example, you will need to manually:

- **Configure your Ray Cluster**: Set up your multi-node environment and manage resource allocation without Anyscale's automation.
- **Manage Dependencies**: Manually install and manage dependencies on each node.
- **Set Up Storage**: Configure your own distributed or shared storage system for model checkpointing.

## Step by Step Guide

In this example, we will demonstrate how to use Ray Train with DeepSpeed to fine-tune a LLM on a multi-GPU (multi-node) environment.
Before start writing a Python script for fine-tuning, install the required dependencies.

```bash
%%bash
pip install torch torchvision
pip install transformers datasets==3.6.0 trl
pip install deepspeed
```

### 1. Import packages

Let's begin our Python script with importing necessary packages and set up a logger.

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

Next, we set up a training dataset and a loader. The following `setup_dataloader` function loads a data set from HuggingFace Hub and returns a Ray's data loader.


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

A data set needs to be *tokenized* by a tokenizer. In many cases, HuggingFace repository offers a tokenizer associated with a model.
We can download the tokenizer with `AutoTokenizer.from_pretrained()`. `load_dataset` is also a HuggingFace library API to download a dataset. We can apply our custom tokenization function with `dataset.map`.
Once dataset is set up, we use `DataLoader`, which is a PyTorch official class to load data from a data set and form a mini-batch.

In addition to these APIs, we also use a Ray's convenient API, `ray.train.torch.prepare_data_loader`. This is useful when you run distributed training using multiple GPUs.
In data parallelism, which is the most common approach for distributed training, we feed a different set of training data on each GPU. 
For this purpose, `ray.train.torch.prepare_data_loader` internally `DistributedSampler` when we use multiple GPUs.
See the [API document](https://docs.ray.io/en/latest/train/api/doc/ray.train.torch.prepare_data_loader.html) for more details.


### 3. Model and optimizer initialization

Next, we will see how we initialize a model and an optimizer.
You can download a model from HuggingFace model hub using `AutoModelForCausalLM.from_pretrained`.
Here we use PyTorch's official implementation of Adam for our optimizer.

`deepspeed.initialize` is an API to enable DeepSpeed. In this example, we pass a model, an optimizer, and a dictionary of configuration items to the API. This API wraps the model to apply various optimization techniques.
The function returns the `DeepSpeedEngine`, which manages the model during distributed training.

```python
def setup_model_and_optimizer(model_name: str, learning_rate: float, ds_config: Dict[str, Any]) -> deepspeed.runtime.engine.DeepSpeedEngine:
    model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    ds_engine, _, _, _ = deepspeed.initialize(
        model=model,
        optimizer=optimizer,
        config=ds_config,
    )
    return ds_engine
```


## 4. Checkpointing and loading


Before we start working on a training loop, let's prepare functions for checkpointing and loading.

Checkpointing is crucial for fault tolerance and resuming training.
The DeepSpeed Engine has `save_checkpoint` API to save a checkpoint. As the DeepSpeed engine has its state (model parameters and optimizer states) in a partitioned state, the API also saves the checkpoint as it is partitioned.

After we make sure all the distributed processes joinining the training finish saving the checkpoint (we use `torch.distributed.barrier` for this purpose), we call `ray.train.report` to report the metrics and saved in a persistent storage.

 
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
```


The `load_checkpoint` function handles restoring the training state by loading a Ray Train `Checkpoint` into the DeepSpeed engine, allowing the training to resume from a previously saved state.

```python
def load_checkpoint(ds_engine: deepspeed.runtime.engine.DeepSpeedEngine, ckpt: ray.train.Checkpoint):
    try:
        with ckpt.as_directory() as checkpoint_dir:
            ds_engine.load_checkpoint(checkpoint_dir)

    except Exception as e:
        raise RuntimeError(f"Checkpoint loading failed: {e}") from e
```


### 5. Training Iteration

In Ray Train, we define a function that orchestrates the entire training process.
This function runs on each process that corresponds to a GPU.

In our example, we first call functions defined above: loading a checkpoint if exists, set up a data loader, and initialize DeepSpeed.
The function then iterates through the specified number of epochs, and for each epoch, it loops over the training data.

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


## 6. Configure DeepSpeed and Launch Trainer

The final step is to launch the training iteration on multiple GPUs. Ray Train offers a simple API designed for the purpose.
We just need to set configurations given by command line arguments and launch `TorchTrainer`.

We use three different types of configurations.
- Training parameters: batch size, learning rate, etc.
- DeepSpeed: Parallelization config, precision, offload, performance tuning parameters like communication buffer size, etc.
- Ray Train: Runtime configurations including storage path, experiment name, etc.

The configurations are passed to `TorchTrainer`, which takes care of launching the training function on multiple GPUs.
Then, we call `trainer.fit()` to start actors, each running the training function.

```python
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
    print(f"Training finished. Result: {result}")


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="MiniLLM/MiniPLM-Qwen-500M")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_epochs", type=int, default=1)
    parser.add_argument("--seq_length", type=int, default=512)
    parser.add_argument("--learning_rate", type=float, default=1e-6)
    parser.add_argument("--zero_stage", type=int, default=3)

    return parser.parse_args()


if __name__ == "__main__":
    main()
```


## Advanced Configurations

DeepSpeed has many other configuration options to tune performance and memory usage.
Here we introduce some of the most commonly used options.
Please refer to the [DeepSpeed documentation](https://www.deepspeed.ai/docs/config-json/) for more details.


### DeepSpeed ZeRO Stages

DeepSpeed ZeRO has three stages, each providing different levels of memory optimization and performance trade-offs.

- **Stage 1**: This stage focuses on optimizer state partitioning. It reduces memory usage by partitioning the optimizer states across data parallel workers. This is the least aggressive stage and is suitable for most models without significant changes.
- **Stage 2**: In addition to optimizer state partitioning, this stage also partitions the gradients. This further reduces memory usage but may introduce some communication overhead. It's a good choice for larger models that can benefit from additional memory savings.
- **Stage 3**: This is the most aggressive stage, which partitions both the optimizer states and the model parameters. It provides the highest memory savings but may require more careful tuning of the training process. This stage is recommended for very large models that cannot fit into the memory of a single GPU.

You can select the desired ZeRO stage by setting the `zero_stage` parameter in the DeepSpeed configuration dictionary passed to `deepspeed.initialize`.

```python
ds_config = {
    "zero_optimization": {
        "stage": 2,  # or 1 or 3
...
    },
}
```


### Mixed Precision Training

Mixed precision training is a technique that uses both 16-bit and 32-bit floating-point types in a single network. This can lead to faster training times and reduced memory usage. DeepSpeed has built-in support for mixed precision training using either FP16 or BF16.

To enable mixed precision training, you can set the `bf16` or `fp16` parameters in the DeepSpeed configuration dictionary. For example:

```python
ds_config = {
    "bf16": {"enabled": True}, # or "fp16": {"enabled": True}
}
```

Note that these options keep the clone of weights/gradients and optimizer states in 32-bit precision to maintain numerical stability.


### CPU Offloading

DeepSpeed supports offloading model states and optimizer states to CPU memory.
Offloading these causes a certain amount of overhead due to data transfer between CPU and GPU, but it significantly reduces GPU memory usage, which can be beneficial when training very large models that do not fit into GPU memory.

To enable CPU offloading, you can set the `offload` parameters in the DeepSpeed configuration dictionary. For example:

```python
ds_config = {
    "offload_param": {
        "device": "cpu",
        "pin_memory": True,
    }
}
```

You can also offload only optimizer states similarly by using the `offload_optimizer` parameter.

```python
ds_config = {
    "offload_optimizer": {
        "device": "cpu",
        "pin_memory": True,
    }
}
```


