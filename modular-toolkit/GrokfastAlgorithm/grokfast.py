from collections import deque
from typing import Dict, Optional, Literal
import torch
import torch.nn as nn


def gradfilter_ma(
    m: nn.Module,
    grads: Optional[Dict[str, deque]] = None,
    window_size: int = 100,
    lamb: float = 5.0,
    filter_type: Literal['mean', 'sum'] = 'mean',
    warmup: bool = True,
    trigger: bool = False, # For ablation study.
) -> Dict[str, deque]:
    if grads is None:
        grads = {n: deque(maxlen=window_size) for n, p in m.named_parameters() if p.requires_grad and p.grad is not None}

    for n, p in m.named_parameters():
        if p.requires_grad and p.grad is not None:
            grads[n].append(p.grad.data.detach()) # .cpu())

            # Modify the gradients.
            if not warmup or len(grads[n]) == window_size and not trigger:
                if filter_type == "mean":
                    avg = sum(grads[n]) / len(grads[n])
                elif filter_type == "sum":
                    avg = sum(grads[n])
                else:
                    raise ValueError(f"Unrecognized filter_type {filter_type}")
                p.grad.data = p.grad.data + avg * lamb

    return grads


def gradfilter_ema(
    m: nn.Module,
    grads: Optional[Dict[str, torch.Tensor]] = None,
    alpha: float = 0.98,
    lamb: float = 2.0,
) -> Dict[str, torch.Tensor]:
    if grads is None:
        grads = {n: p.grad.data.detach() for n, p in m.named_parameters() if p.requires_grad and p.grad is not None}

    for n, p in m.named_parameters():
        if p.requires_grad and p.grad is not None:
            grads[n] = grads[n] * alpha + p.grad.data.detach() * (1 - alpha)
            p.grad.data = p.grad.data + grads[n] * lamb

    return grads



def self_adoptive(config, loss_recoder, acc_recoder):
    if len(loss_recoder) < 2 or len(acc_recoder) < 2:
        return
    if loss_recoder[-1] < loss_recoder[-2] and acc_recoder[-1] < acc_recoder[-2]:    #如果loss减小 acc也减小  alpha增大， lamb增大
        config['grokfast']['lamb'] = config['grokfast']['lamb'] * (1 + config['grokfast']['self_adoptive_rate']) if config['grokfast']['lamb'] * (1 + config['grokfast']['self_adoptive_rate']) <= 5 else 5
        config['grokfast']['alpha'] = config['grokfast']['alpha'] * (1 + config['grokfast']['self_adoptive_rate']) if config['grokfast']['alpha'] *(1 + config['grokfast']['self_adoptive_rate']) <= 0.99 else 0.99
    elif loss_recoder[-1] < loss_recoder[-2] and acc_recoder[-1] > acc_recoder[-2]:   #如果loss减小 acc增大  alpha减小， lamb增大
        config['grokfast']['alpha'] = config['grokfast']['alpha'] * (1 - config['grokfast']['self_adoptive_rate']) if config['grokfast']['alpha'] *(1 + config['grokfast']['self_adoptive_rate']) >= 0.1 else 0.1
        config['grokfast']['lamb'] = config['grokfast']['lamb'] * (1 + config['grokfast']['self_adoptive_rate']) if config['grokfast']['lamb'] * (1 + config['grokfast']['self_adoptive_rate']) <= 5 else 5
    elif loss_recoder[-1] > loss_recoder[-2] and acc_recoder[-1] > acc_recoder[-2]:   #如果loss增大 acc增大  alpha增大， lamb减小
        config['grokfast']['alpha'] = config['grokfast']['alpha'] * (1 + config['grokfast']['self_adoptive_rate']) if config['grokfast']['alpha'] *(1 + config['grokfast']['self_adoptive_rate']) <= 0.99 else 0.99
        config['grokfast']['lamb'] = config['grokfast']['lamb'] * (1 - config['grokfast']['self_adoptive_rate']) if config['grokfast']['lamb'] * (1 + config['grokfast']['self_adoptive_rate']) >= 1 else 1
    elif loss_recoder[-1] > loss_recoder[-2] and acc_recoder[-1] < acc_recoder[-2]:   #如果loss增大 acc减小  alpha减小， lamb减小
        config['grokfast']['alpha'] = config['grokfast']['alpha'] * (1 - config['grokfast']['self_adoptive_rate']) if config['grokfast']['alpha'] *(1 + config['grokfast']['self_adoptive_rate']) >= 0.1 else 0.1
        config['grokfast']['lamb'] = config['grokfast']['lamb'] * (1 - config['grokfast']['self_adoptive_rate']) if config['grokfast']['lamb'] * (1 + config['grokfast']['self_adoptive_rate']) >= 1 else 1
    else:
        pass