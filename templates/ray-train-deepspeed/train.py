import os
os.environ["RAY_TRAIN_V2_ENABLED"] = "1"

import ray
import ray.train
import ray.train.torch
from ray.train.torch import TorchTrainer
from ray.train import ScalingConfig, RunConfig, Checkpoint

import torch
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
from torch.utils.data import DataLoader

from torchvision.models import VisionTransformer
from torchvision.datasets import FashionMNIST
from torchvision.transforms import ToTensor, Normalize, Compose

import tempfile
import uuid
import logging
logger = logging.getLogger(__name__)


def init_model() -> torch.nn.Module:
    model = VisionTransformer(
        image_size=28,
        patch_size=7,
        num_layers=4,
        num_heads=2,
        hidden_dim=64,
        mlp_dim=128,
        num_classes=10,
    )
    model.conv_proj = torch.nn.Conv2d(
        in_channels=1,
        out_channels=64,
        kernel_size=7,
        stride=7,
    )
    return model


def train_loop(config: dict):
    model = init_model()
    device = ray.train.torch.get_device()
    torch.cuda.set_device(device)
    model.to(device)

    optimizer = Adam(model.parameters(), lr=config.get("learning_rate", 1e-3))
    criterion = CrossEntropyLoss()

    transform = Compose([ToTensor(), Normalize((0.5,), (0.5,))])
    data_dir = os.path.join(tempfile.gettempdir(), "data")
    train_ds = FashionMNIST(root=data_dir, train=True, download=True, transform=transform)

    dataloader = DataLoader(train_ds, batch_size=config.get("batch_size", 128), shuffle=True)
    dataloader = ray.train.torch.prepare_data_loader(dataloader)

    ckpt = ray.train.get_checkpoint()
    if ckpt:
        with ckpt.as_directory() as ckpt_dir:
            state = torch.load(os.path.join(ckpt_dir, "state.pt"), map_location="cpu")
            model.load_state_dict(state["model"])
            optimizer.load_state_dict(state["optim"])

    world_rank = ray.train.get_context().get_world_rank()

    epochs = config.get("epochs", 5)
    step = 0
    for epoch in range(epochs):
        running_loss = 0.0
        batches = 0
        for images, labels in dataloader:
            optimizer.zero_grad(set_to_none=True)
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            batches += 1
            step += 1

        avg_loss = running_loss / max(1, batches)

        with tempfile.TemporaryDirectory() as tmp:
            torch.save({
                "model": model.state_dict(),
                "optim": optimizer.state_dict(),
                "epoch": epoch,
                "step": step,
            }, os.path.join(tmp, "state.pt"))
            ray.train.report({"loss": avg_loss, "epoch": epoch}, checkpoint=Checkpoint.from_directory(tmp))

        if world_rank == 0:
            print({"loss": avg_loss, "epoch": epoch})


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
    run_config=run_config
)

result = trainer.fit()
print("Training finished", result)
