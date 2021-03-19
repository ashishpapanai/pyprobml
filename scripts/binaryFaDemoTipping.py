import numpy as np
import math
import matplotlib.pyplot as plt
import os

if os.path.isdir('scripts'):
    os.chdir('./scripts')

np.random.seed(0)
D = 16
K = 3
proto = np.random.rand(D, K) < 0.5
M = 50
#source = np.concatenate(1*np.ones((1, M)), 2 * np.ones((1, M)), 3*np.ones((1, M)))
source = np.concatenate((np.concatenate(
    (0*np.ones((1, M)), 1 * np.ones((1, M))), axis=1), 2*np.ones((1, M))), axis=1)

N = source.size
dataClean = np.zeros((N, D))
for n in range(0, N):
    src = source[:, n]
    dataClean[n, :] = np.transpose(proto[:, src.astype(int)])

noiseLevel = 0.05
flipMask = np.random.rand(N, D) < noiseLevel
dataNoisy = dataClean
dataNoisy[flipMask] = 1-dataClean[flipMask]
dataMissing = dataClean
dataMissing[flipMask] = math.nan

plt.imshow(dataNoisy,  cmap='gray', interpolation='nearest', aspect='auto')
plt.title('noisy binary data')
plt.savefig('../figures/binaryPCAinput')
plt.show()

plt.imshow(dataClean,  cmap='gray', interpolation='nearest', aspect='auto')
plt.title('hidden truth')
plt.savefig('../figures/binaryPCAhidden')
plt.show()
