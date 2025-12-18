import numpy as np
import matplotlib.pyplot as plt

data = (
    np.fromfile("/media/lonqi/PS2000/rat_01_part5/20_30/0001_0.0000_0.raw", dtype=np.uint32)
        .astype(float).reshape(128,384)
    )
data = data / 10
k = 1