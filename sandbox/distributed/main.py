from ddp_trainer import *
import torch.multiprocessing as mp
from torch.utils.data import Dataset, TensorDataset
from torch import nn
from fashion import training_data, test_data, NeuralNetwork

def main(rank, world_size):
    ddp_setup(rank, world_size)

    training_loader = prepare_dataloader(training_data, batch_size=64)
    
    trainer = Trainer(model=net,
                      train_data=dl,
                      optimizer=torch.optim.Adam(net.parameters(), lr=0.01),
                      criterion=nn.MSELoss(),
                      save_every=10,
                      id=rank)
    
    trainer.train(epochs=100)

    ddp_exit()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Distributed training job")
    parser.add_argument("--nprocs", default=12, type=int, help="How many threads (default: 12)")
    args = parser.parse_args()
    world_size = args.nprocs
    mp.spawn(main, args=(world_size, ), nprocs=world_size)
