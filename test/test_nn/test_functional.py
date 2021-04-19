#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
import numpy as np
import src.nn.functional as F


def test_01():
    x1 = np.array([0, 2, 1, 4, 5, 3])
    print("x1:\n", x1)

    ret1 = F.one_hot(x1)
    print("ret1:\n", ret1)

    ret2 = F.one_hot(x1, 8)
    print("ret2\n", ret2)
