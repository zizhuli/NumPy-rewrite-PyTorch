#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

import numpy as np


def one_hot(index: np.ndarray, num_classes: int = -1) -> np.ndarray:
    if num_classes == -1:
        return np.eye(np.max(index) + 1, dtype=np.int64)[index]
    else:
        return np.eye(num_classes, dtype=np.int64)[index]
