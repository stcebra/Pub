import torch
import torch.nn as nn
from Bilateral_CostVolume import BilateralCostVolume
import numpy as np

dd = np.cumsum([128, 128, 96, 64, 32])
print(dd)
