import numpy as np
SAVE_DIR = '/home/gaoha/PQ-Transformer/statistics_tbw'
stats = []
for i in range(157):
    path = f"{SAVE_DIR}/distance-{i}.npy"
    stats.append(np.load(path))
ret = np.concatenate(stats)

import matplotlib.pyplot as plt
plt.cla()
plt.figure(dpi=800)
plt.hist((ret)[np.abs(ret)<5], bins=1000, color='g', alpha=0.5)
plt.savefig('out_tbw.png')