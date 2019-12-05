import numpy as np
import matplotlib.pyplot as plt
from itertools import cycle, islice
from sklearn.cluster import KMeans
from sklearn import datasets
def gendata(n_samples):
    x,y = datasets.make_circles(n_samples,factor=0.6,noise=0.03)
    return x,y

#compute the AdjaceincyMatrix
def euclidDistance(x1, x2, sqrt_flag=False):
    res = np.sum((x1-x2)**2)
    if sqrt_flag:
        res = np.sqrt(res)
    return res
def calEuclidDistanceMatrix(X):
    X = np.array(X)
    S = np.zeros((len(X), len(X)))
    for i in range(len(X)):
        for j in range(i+1, len(X)):
            S[i][j] = 1.0 * euclidDistance(X[i], X[j])
            S[j][i] = S[i][j]
    return S
def AdjaceincyMatrix(S,k,sigma=1.0):
    N = len(S)
    W = np.zeros((N,N))

    for i in range(N):
        dist_with_index = zip(S[i], range(N))
        dist_with_index = sorted(dist_with_index, key=lambda x:x[0])
        #Consider xi's k nearest neighbors
        neighbours_id = [dist_with_index[m][1] for m in range(k+1)]

        for j in neighbours_id: 
            # xj is ones of the neighbours of xi's 
            W[i][j] = np.exp(-S[i][j]/2/sigma/sigma)
            W[j][i] = W[i][j]

    return W

def LaplacecianMatrix(adjaceincyMatrix):
    #compute the D Matrix:D=diag(d1,...,dn)
    # di=w[i][1]+w[i][2]+...
    D = np.diag(np.sum(adjaceincyMatrix,axis=1))

    #compute the Laplacian Matrix:L=D-W
    LaplacecianMatrix = D - adjaceincyMatrix

    #normalize the LaplacecianMatrix
    A = np.sum(adjaceincyMatrix,axis=1)
    nor_lapmatrix = np.dot(np.dot(np.diag(1.0 / (A ** (0.5))),LaplacecianMatrix),np.diag(1.0 / (A ** (0.5))))
    return nor_lapmatrix

def spKmeans(V):
    sp_kmeans = KMeans(n_clusters=2).fit(V)
    return sp_kmeans.labels_

def plot(X, sp_y, km_y):
    colors = np.array(list(islice(cycle(['#377eb8', '#ff7f00', '#4daf4a',
                                                '#f781bf', '#a65628', '#984ea3',
                                                '#999999', '#e41a1c', '#dede00']),
                                        int(max(km_y) + 1))))
    plt.subplot(121)
    plt.scatter(X[:,0], X[:,1], s=10, color=colors[sp_y])
    plt.title("Spectral Clustering")
    plt.subplot(122)
    plt.scatter(X[:,0], X[:,1], s=10, color=colors[km_y])
    plt.title("Kmeans Clustering")
    plt.show()

np.random.seed(123)
data, label = gendata(n_samples=1000)
Similarity = calEuclidDistanceMatrix(data)
Adjacent = AdjaceincyMatrix(Similarity, k=5)
Laplacian = LaplacecianMatrix(Adjacent)

x, Z = np.linalg.eig(Laplacian)
x = zip(x, range(len(x)))
x = sorted(x, key=lambda x:x[0])
V = np.vstack([Z[:,i] for (v, i) in x[:1000]]).T
#If the procedure runs for too long, you can properly adjust the scale of n_sample and the value of k

#SpectralClustering with kmeans
sp_kmeans = KMeans(n_clusters=2).fit(V)
#kmeans
kmeans = KMeans(n_clusters=2).fit(data)
plot(data, sp_kmeans.labels_, kmeans.labels_)



#Reference Songdark's SpectralClustering