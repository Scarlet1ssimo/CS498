### Q3: allreduce###
### please implement ring_allreduce method, using  pytorch's dist method is not allowed###

from torch._utils import _flatten_dense_tensors, _unflatten_dense_tensors
import torch
import torch.distributed as dist


def reduce_scatter(chunks, tmp, send_idx, recv_idx, left, right):
    r = dist.irecv(tmp, src=left)
    s = dist.isend(chunks[send_idx], dst=right)
    r.wait()
    chunks[recv_idx] += tmp
    s.wait()


def all_gather(chunks, tmp, send_idx, recv_idx, left, right):
    r = dist.irecv(chunks[recv_idx], src=left)
    s = dist.isend(chunks[send_idx], dst=right)
    r.wait()
    chunks[recv_idx] = tmp
    s.wait()


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
    for i in range(0, world-1):
        recv_idx = (rank-i-1+world) % world
        send_idx = (recv_idx+1) % world
        print(
            f"reduce_scatter step {i} for device {rank}: chunk[{send_idx}] send to {right}, chunk[{recv_idx}] recv from {left}")
        reduce_scatter(chunks, tmp, send_idx, recv_idx, left, right)

    for i in range(0, world-1):
        recv_idx = (rank-i+world) % world
        send_idx = (recv_idx+1) % world
        print(
            f"all_gather step {i} for device {rank}: chunk[{send_idx}] send to {right}, chunk[{recv_idx}] recv from {left}")
        all_gather(chunks, tmp, send_idx, recv_idx, left, right)

    # stitch & unpad
    flat /= world
    tensor.view(-1).copy_(flat[:n])
    return
