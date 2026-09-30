"""Set PyTorch threads before importing CEDDAR, then use its original CLI."""
import os
import runpy
import sys
import torch

threads = int(os.environ['CPU_THREADS'])
torch.set_num_threads(threads)
torch.set_num_interop_threads(1)
print(f'PyTorch {torch.__version__}: intra-op={torch.get_num_threads()}, '
      f'inter-op={torch.get_num_interop_threads()}', flush=True)
sys.argv = ['repro.sigma_star', *sys.argv[1:]]
runpy.run_module('repro.sigma_star', run_name='__main__')
