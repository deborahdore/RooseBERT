import gc

import torch


def clean():
    if torch.distributed.is_initialized():
        torch.distributed.barrier()
        torch.distributed.destroy_process_group()
    gc.collect()
    torch.cuda.empty_cache()
