import numpy as np
import random
from sklearn.datasets.samples_generator import make_blobs
import matplotlib.pyplot as plt

def euclidean_dist(a, b):
    dist = np.linalg.norm(np.array(a) - np.array(b))
    return dist

def gaussian_kernel(euclidean_dist, bandwidth):
    kernel = (1 / (bandwidth * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((euclidean_dist / bandwidth)) ** 2)
    return kernel

class MeanShift(object):
    def __init__(self, kernel=gaussian_kernel):
        self.kernel = kernel

    def shift_point(self, point, points, kernel_bandwidth):
        shift_x = float(0)
        shift_y = float(0)
        scale_factor = float(0)
        for p_temp in points:
            # numerator
            dist = euclidean_dist(point, p_temp)
            weight = self.kernel(dist, kernel_bandwidth)
            shift_x += p_temp[0] * weight
            shift_y += p_temp[1] * weight
            # denominator
            scale_factor += weight
        shift_x = shift_x / scale_factor
        shift_y = shift_y / scale_factor
        return [shift_x, shift_y]

    def cluster_points(self, points):
        cluster_ids = []
        cluster_idx = 0
        cluster_centers = []
        for i, point in enumerate(points):
            if(len(cluster_ids) == 0):
                cluster_ids.append(cluster_idx)
                cluster_centers.append(point)
                cluster_idx += 1
            else:
                for center in cluster_centers:
                    dist = euclidean_dist(point, center)
                    if(dist < Cluster_Threshold):
                        cluster_ids.append(cluster_centers.index(center))
                if(len(cluster_ids) < i + 1):
                    cluster_ids.append(cluster_idx)
                    cluster_centers.append(point)
                    cluster_idx += 1
        return cluster_ids

    def fit(self, points, kernel_bandwidth):
        shift_points = np.array(points)
        shifting = [True] * points.shape[0]
        while True:
            max_dist = 0
            for i in range(0, len(shift_points)):
                if not shifting[i]:
                    continue
                p_shift_init = shift_points[i].copy()
                shift_points[i] = self.shift_point(shift_points[i], points, kernel_bandwidth)
                dist = euclidean_dist(shift_points[i], p_shift_init)
                max_dist = max(max_dist, dist)
                shifting[i] = dist > Stop_Threshold
            if(max_dist < Stop_Threshold):
                break
        cluster_ids = self.cluster_points(shift_points.tolist())
        return shift_points, cluster_ids


def colors(n):
  ret = []
  for i in range(n):
    ret.append((random.uniform(0, 1), random.uniform(0, 1), random.uniform(0, 1)))
  return ret

def main():
    centers = [[-1, -1], [-1, 1], [1, -1], [1, 1]]
    X, _ = make_blobs(n_samples=200, centers=centers, cluster_std=0.3)
    mean_shifter = MeanShift()
    _, mean_shift_result = mean_shifter.fit(X, kernel_bandwidth=0.3)
    print('input: {}'.format(X))
    print('assined clusters: {}'.format(mean_shift_result))
    color = colors(np.unique(mean_shift_result).size)
    for i in range(len(mean_shift_result)):
        plt.scatter(X[i, 0], X[i, 1], color = color[mean_shift_result[i]])
    for i, j in centers:
        plt.scatter(i, j, s=100, c='black', marker='*')
    plt.show()

Stop_Threshold = 1e-5
Cluster_Threshold = 1e-1

if __name__ == '__main__':
    main()