import os
os.environ["CUDA_VISIBLE_DEVICES"]="1"
print("imported os")
import argparse
import csv
print("imported argpas , csv")
from pathlib import Path
print("imported pathlib")
from typing import List, Optional, Union
print("imported typing stuff")
import numpy as np
print("imported numpy")
import torch
print("imported torch")
import torch.nn.functional as F
print("imported torch.nn.functional")
from evo2 import Evo2
