import torch
import numpy as np
from copy import copy

# label_numpy.shape = Tx1xHxW
# target, out.size = Tx3xHxW


def person_paste(label, target, gen):

    # 人物領域抜き取り
    person_one = label.clone()
    person_zero = label.clone()

    person_one[person_one == 0] = 255
    person_one[person_one < 255] = 0
    person_one[person_one == 255] = 1
    person_zero[person_zero != 0] = 1

    person_domain = person_one * target
    other_domain = person_zero * gen

    out = person_domain + other_domain

    # out = torch.from_numpy(out)

    return out
