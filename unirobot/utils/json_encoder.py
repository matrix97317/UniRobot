# -*- coding: utf-8 -*-
"""Json encoder."""
import json

import numpy as np
import torch


class URJsonEncoder(json.JSONEncoder):
    """UniRobot Encoder."""

    def np_encoder(self, obj):
        """Numpy encoder."""
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)

    def default(self, obj):  # pylint: disable=arguments-renamed.
        """Override default."""
        if isinstance(obj, torch.Tensor):
            obj = obj.detach().cpu().numpy()
            return self.np_encoder(obj)

        return self.np_encoder(obj)
