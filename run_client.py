from argparse import ArgumentParser

import torch
import torch.nn as nn
from hivemind import RemoteMixtureOfExperts, DHT


class TrainerModel(nn.Module):
    def __init__(self, expert_dim, grid_size, dht, output_dim, num_moe_blocks=2):
        super().__init__()
        self.mixture = nn.Sequential(
            *[RemoteMixtureOfExperts(in_features=expert_dim, grid_size=grid_size, dht=dht,
                                     k_best=4, forward_timeout=10, backward_timeout=1, timeout_after_k_min=1,
                                     uid_prefix='expert.')
              for _ in range(num_moe_blocks)])
        self.output_proj = nn.Linear(expert_dim, output_dim)

    def forward(self, x):
        return self.mixture(x)


def main(args):
    init_peers = [args.rendezvous] if args.rendezvous is not None else []
    dht = DHT(initial_peers=init_peers, wait_timeout=5, start=True, listen=False)
    trainer = TrainerModel(args.hidden_dim, [args.grid_size for _ in range(args.grid_dimensions)], dht, 4)
    for i in range(999):
        print(f'{i} forwarding')
        output = trainer(torch.randn(32, args.hidden_dim))


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--grid-size', '-g', type=int, default=100)
    parser.add_argument('--grid-dimensions', '-d', type=int, default=3)
    parser.add_argument('--hidden_dim', type=int, default=1024, required=False, help='main dimension for expert_cls')
    parser.add_argument('--rendezvous')
    args = parser.parse_args()
    main(args)
