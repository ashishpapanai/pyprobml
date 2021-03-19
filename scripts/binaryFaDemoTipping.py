import numpy as np
import math
import matplotlib.pyplot as plt
import os
from sklearn.decomposition import FactorAnalysis

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
    check = (proto[:, src.astype(int)]).reshape(16, )
    dataClean[n, :] = check


noiseLevel = 0.05
flipMask = np.random.rand(N, D) < noiseLevel
dataNoisy = dataClean

for i in range(0, N):
    for j in range(0, D):
        if flipMask[i][j]:
            dataNoisy[i][j] = 1 - dataClean[i][j]
'''
dataMissing = dataClean
for i in range(0, N):
    for j in range(0, D):
        if flipMask[i][j]:
            dataMissing[i][j] = math.nan'''

# print(dataNoisy)

plt.imshow(dataNoisy,  cmap='gray', interpolation='nearest', aspect='auto')
plt.title('noisy binary data')
plt.savefig('../figures/binaryPCAinput')
plt.show()

plt.imshow(dataClean,  cmap='gray', interpolation='nearest', aspect='auto')
plt.title('hidden truth')
plt.savefig('../figures/binaryPCAhidden')
plt.show()

model = FactorAnalysis(n_components=2, max_iter=10)
loglikHist = model.fit_transform(X=dataNoisy)
# print(type(loglikHist))
plt.plot(loglikHist)
plt.title('(lower bound on) loglik vs iter for EM')
plt.show()

muPost = model.transform(dataNoisy)
muPost = np.transpose(muPost)
symbols = ['ro', 'gs', 'k*']
for k in range(0, K):
    ndx = (source == k)
    ndx = ndx.reshape(150,)
    occurrences = np.count_nonzero(ndx == 1)
    a = np.zeros((occurrences,))
    b = np.zeros((occurrences,))
    x = 0
    for i in range(0, 150):
        if ndx[i]:
            a[x] = muPost[0, i]
            b[x] = muPost[1, i]
            x += 1
    plt.plot(a, b, symbols[k])

plt.title('latent embedding')
plt.savefig('../figures/binaryPCAembedding')
plt.show()

