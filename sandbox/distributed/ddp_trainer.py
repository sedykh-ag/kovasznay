import os
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
import torch.distributed as dist


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
        world_size: int,
        save_every: int,
    ) -> None:
        self.id = id
        self.world_size = world_size
        self.model = model
        self.train_data = train_data
        self.optimizer = optimizer
        self.criterion = criterion
        self.save_every = save_every
        self.epochs_run = 0

        self.model = DDP(self.model, device_ids=None)
        
    def _load_snapshot(self, snapshot_path):
        loc = "cpu"
        snapshot = torch.load(snapshot_path, map_location=loc)
        self.model.load_state_dict(snapshot["MODEL_STATE"])
        self.epochs_run = snapshot["EPOCHS_RUN"]

        print(f"Resuming training from snapshot at Epoch {self.epochs_run}")

    def _save_snapshot(self, epoch, snapshot_path="snapshots/"):
        os.makedirs(snapshot_path, exist_ok=True)
        if self.id == 0:
            snapshot = {
                "MODEL_STATE": self.model.module.state_dict(),
                "EPOCHS_RUN": epoch,
            }
            PATH = os.path.join(snapshot_path, "ckpt.pt")
            torch.save(snapshot, PATH)
            print(f"[CPU {self.id}] epoch {epoch} | Saved snapshot at {PATH}")


    def _run_batch(self, source, target):
        self.optimizer.zero_grad()
        output = self.model(source)
        loss = self.criterion(output, target)
        loss.backward()
        self.optimizer.step()
        return loss.item()
    
    def _run_epoch(self, epoch):
        batch_size = len(next(iter(self.train_data))[0])
        n_batches = len(self.train_data)

        self.train_data.sampler.set_epoch(epoch)
        loss_epoch = 0.0
        for i, data in enumerate(self.train_data):
            source, targets = data
            loss_epoch += self._run_batch(source, targets)
            if self.id == 0:
                print(f"[CPU {self.id}] epoch: {epoch} | processed batch {i} | batchsize: {batch_size}")
        loss_epoch = torch.tensor(loss_epoch / n_batches, dtype=float)
        #print(f"[CPU {self.id}] epoch: {epoch} | batchsize: {batch_size}")
        # print(f"[CPU {self.id}] epoch: {epoch} | batchsize: {batch_size} | n_batches: {n_batches} | loss: {loss_epoch.item()}")

        # get total loss via all_reduce
        dist.all_reduce(loss_epoch, op=dist.ReduceOp.SUM)
        loss_total = loss_epoch.item() / self.world_size
        if (self.id == 0):
            print(f"[CPU {self.id}] total_loss: {loss_total}")
        
        if epoch != 0 and epoch % self.save_every == 0:
            self._save_snapshot(epoch=epoch)


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