# NoahRGB 09/2026
#
# a file containing misc util functions
#

import numpy as np
import torch
import random

def seed(s: int):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(s)
