### Q3: allreduce###
### please implement ring_allreduce method, using  pytorch's dist method is not allowed###

from torch._utils import _flatten_dense_tensors, _unflatten_dense_tensors
import torch
import torch.distributed as dist


def reduce_scatter(chunks, tmp, world, rank, left, right):
    #                                                                   #
    #                                                                   #
    # your code here: follow slides instruction: do counter-clockwise iteration
    #                                                                   #
    #                                                                   #
    return


def all_gather(chunks, tmp, current, world, rank, left, right):
    #                                                                   #
    #                                                                   #
    # your code here: follow slides instruction: do counter-clockwise iteration
    #                                                                   #
    #                                                                   #
    return


def ring_allreduce_(tensor: torch.Tensor, world_size=None, rankid=None):
    """In-place ring all-reduce (SUM, optional average) using isend/irecv."""
    world = world_size
    if world == 1:
        return tensor
    rank = rankid
    left, right = (rank - 1) % world, (rank + 1) % world

    # following steps try to fill blank to the tensor so that final tensor can be divided to 3 chunks evenly
    flat = tensor.contiguous().view(-1)
    n = flat.numel()
    chunk = (n + world - 1) // world

    padded_flat = torch.zeros(
        chunk * world, dtype=flat.dtype, device=flat.device)
    padded_flat[:n] = flat
    chunks = [padded_flat[i*chunk:(i+1)*chunk] for i in range(world)]

    tmp = torch.zeros_like(chunks[0])
    for i in range(1, world):
        send_idx = (rank-i-1+world) % world
        recv_idx = (send_idx+1) % world
        reduce_scatter(chunks, tmp, world, rank, left, right)
        chunks[send_idx] = tmp

    for i in range(0, world-1):
        all_gather(chunks, tmp, i, world, rank, left, right)
        chunks[(world+rank-i) % world] = tmp

    # stitch & unpad
    flat /= world
    tensor.view(-1).copy_(flat[:n])
    return
