import argparse

import torch
from julius import ResampleFrac
from torch.utils.benchmark import Compare, Timer

from kazane import Upsample

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cpu')

    args = parser.parse_args()
    device = args.device

    batch = 16
    sr = 44100
    up_ratios = [2, 3, 5, 7]
    duration = 10
    zeros = 24
    x = torch.randn(batch, int(sr * duration), device=device)

    num_threads = torch.get_num_threads()
    print(f'Benchmarking on {num_threads} threads')

    results = []
    for q in up_ratios:
        label = 'Up sample'
        sub_label = f'rate: {q}'

        results.append(Timer(
            stmt='m(x)',
            setup='',
            globals={'x': x, 'm': ResampleFrac(sr, sr * q, zeros).to(device)},
            num_threads=num_threads,
            label=label,
            sub_label=sub_label,
            description='julius',
        ).blocked_autorange(min_run_time=1))
        results.append(Timer(
            stmt='m(x)',
            setup='',
            globals={'x': x, 'm': Upsample(
                q, zeros).to(device)},
            num_threads=num_threads,
            label=label,
            sub_label=sub_label,
            description='kazane',
        ).blocked_autorange(min_run_time=1))

    compare = Compare(results)
    compare.print()
