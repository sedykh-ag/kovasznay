import os
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group

import pennylane as qml


def ddp_setup():
    init_process_group(backend="gloo")


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        train_data: Dataset,
        optimizer: torch.optim.Optimizer,
        criterion: nn.modules.loss._Loss,
        save_every: int,
        snapshot_path: str = "snapshots/ckpt.pt",
    ) -> None:
        self.id = int(os.environ["LOCAL_RANK"])
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
        loss.backward(0)
        self.optimizer.step()
    
    def _run_epoch(self, epoch):
        batch_size = len(next(iter(self.train_data))[0])
        print(f"[CPU {self.id}] Epoch: {epoch} | Batchsize: {batch_size} | Steps: {len(self.train_data)}")
        self.train_data.samplet.set_epoch(epoch)
        for source, targets in self.train_data:
            self._run_batch(source, targets)

    def _save_snapshot(self, epoch):
        snapshot = {
            "MODEL_STATE_DICT": self.model.module.state_dict(),
            "EPOCHS_RUN": epoch,
        }
        torch.save(snapshot, self.snapshot_path)
        print(f"Epoch {epoch} | Training snapshot saved at {self.snapshot_path}")

    def train(self, epochs: int):
        for epoch in range(self.epochs_run, epochs):
            self._run_epoch(epoch)
            if self.id == 0 and epoch % self.save_every == 0:
                self._save_snapshot(epoch)


def prepare_dataloader(dataset: Dataset, batch_size: int):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        sampler=DistributedSampler(dataset),
    )