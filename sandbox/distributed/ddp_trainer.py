import os
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group


def ddp_setup(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    init_process_group(backend="gloo", rank=rank, world_size=world_size)

def ddp_exit():
    destroy_process_group()

class Trainer:
    def __init__(
        self,
        model: nn.Module,
        train_data: DataLoader,
        optimizer: torch.optim.Optimizer,
        criterion: nn.modules.loss._Loss,
        id: int,
        save_every: int,
        snapshot_path: str = "snapshots/ckpt.pt",
    ) -> None:
        self.id = id
        self.model = model
        self.train_data = train_data
        self.optimizer = optimizer
        self.criterion = criterion
        self.save_every = save_every
        self.epochs_run = 0
        self.snapshot_path = snapshot_path
        if os.path.exists(snapshot_path):
            print("Loading snapshot")
            self._load_snapshot(snapshot_path)
        
        self.model = DDP(self.model, device_ids=None)
        
    def _load_snapshot(self, snapshot_path):
        loc = "cpu"
        snapshot = torch.load(snapshot_path, map_location=loc)
        self.model.load_state_dict(snapshot["MODEL_STATE_DICT"])
        self.epochs_run = snapshot["EPOCHS_RUN"]

        print(f"Resuming training from snapshot at Epoch {self.epochs_run}")

    def _run_batch(self, source, target):
        self.optimizer.zero_grad()
        output = self.model(source)
        loss = self.criterion(output, target)
        loss.backward()
        self.optimizer.step()
    
    def _run_epoch(self, epoch):
        batch_size = len(next(iter(self.train_data))[0])
        print(f"[CPU {self.id}] Epoch: {epoch} | Batchsize: {batch_size} | Steps: {len(self.train_data)}")
        self.train_data.sampler.set_epoch(epoch)
        for source, targets in self.train_data:
            self._run_batch(source, targets)

    def train(self, epochs: int):
        for epoch in range(self.epochs_run, epochs):
            self._run_epoch(epoch)


def prepare_dataloader(dataset: Dataset, batch_size: int):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        sampler=DistributedSampler(dataset),
    )