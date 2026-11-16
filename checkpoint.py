"""Checkpoint loading across PyTorch versions without unsafe modern fallbacks."""
import inspect

import torch


def load_checkpoint(path, map_location='cpu', trusted=False):
    """Use restricted loading when available; require trust on older PyTorch.

    ``trusted=True`` permits legacy pickle loading only when ``torch.load``
    lacks the explicit ``weights_only`` parameter. It never disables restricted
    loading on newer versions, even if a checkpoint fails to deserialize.
    """
    if 'weights_only' in inspect.signature(torch.load).parameters:
        return torch.load(path, map_location=map_location, weights_only=True)
    if not trusted:
        raise ValueError(
            'This PyTorch version has no weights_only loader. Load only a '
            'checkpoint you created or obtained from a trusted source; pass '
            '--trust-checkpoint to acknowledge this for --weights/--init-weights.'
        )
    return torch.load(path, map_location=map_location)
