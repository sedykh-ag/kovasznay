from ddp_trainer import *
import torch.multiprocessing as mp
from torch.utils.data import Dataset, TensorDataset
from torch import nn
from utils_quantum import *

torch.manual_seed(0)
np.random.seed(0)

def main(rank, world_size):
    ddp_setup(rank, world_size)

    net = EncNet()
    training_loader = prepare_dataloader(get_dataset(), batch_size=10)
    
    trainer = Trainer(model=net,
                      train_data=training_loader,
                      optimizer=torch.optim.Adam(net.parameters(), lr=0.01),
                      criterion=nn.MSELoss(),
                      save_every=2,
                      id=rank,
                      world_size=world_size,)
    
    trainer.train(epochs=10)

    ddp_exit()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Distributed training job")
    parser.add_argument("--nprocs", default=12, type=int, help="How many threads (default: 12)")
    args = parser.parse_args()
    world_size = args.nprocs
    mp.spawn(main, args=(world_size, ), nprocs=world_size)
