import numpy as num
import random
import matplotlib.pyplot as plot
#we've imported the libraries ive used

#the following method loads a file and reads the data
def load_data(fname):
    features = []
    with open(fname) as F:
        for line in F:
            p = line.strip().split(' ')
            features.append(num.array(p[1:], dtype=float))#same logic as mentioned in other files/
    return num.array(features)

def ComputeDistance(a, b):
    return num.linalg.norm(a - b)#euclidean distance

def kmeans(x, k, maxIter=100):
    random.seed(20)
    indices = random.sample(range(x.shape[0]), k)
    centroids = x[indices, :]
    for _ in range(maxIter):
        clusters = {i: [] for i in range(k)}
        for point in x:
            distances = [ComputeDistance(point, centroid) for centroid in centroids]
            minDistanceIndex = distances.index(min(distances))
            clusters[minDistanceIndex].append(point)
        
        new_centroids = num.array([num.mean(clusters[i], axis=0) if clusters[i] else centroids[i] for i in range(k)])
        if num.allclose(centroids, new_centroids, atol=1e-6):
            break
        centroids = new_centroids
    return clusters, centroids

def bisecting_kmeans(x, num_clusters):#handles the heirarchial splitting process
    clusters = {0: x}  #starting with one cluster containing all points
    centroids = {0: num.mean(x, axis=0)}
    while len(clusters) < num_clusters:#biggest cluster gets split
        largest_cluster_key = max(clusters, key=lambda k: len(clusters[k]))
        cluster_to_split = clusters.pop(largest_cluster_key)
        _, new_centroids = kmeans(cluster_to_split, 2)#splitting

        for i, centroid in enumerate(new_centroids):
            new_cluster = cluster_to_split[[ComputeDistance(point, centroid) == min([ComputeDistance(point, c) for c in new_centroids]) for point in cluster_to_split]]
            clusters[len(centroids)] = new_cluster
            centroids[len(centroids)] = centroid
    return clusters, centroids

def compute_silhouette(x, clusters, centroids):#the method calculates the silhouette co-efficient for each clustering configuration 
    silhouette_scores = []
    for cluster_key, cluster in clusters.items():
        if len(cluster) < 2:
            continue
        intra_distances = [ComputeDistance(point, centroids[cluster_key]) for point in cluster]
        a = num.mean(intra_distances)
        inter_cluster_distances = []
        for other_key, other_cluster in clusters.items():
            if cluster_key != other_key:
                inter_distances = [ComputeDistance(point, centroids[other_key]) for point in cluster]
                inter_cluster_distances.append(num.mean(inter_distances))
        if inter_cluster_distances:
            b = min(inter_cluster_distances)
            score = (b - a) / max(a, b)
            silhouette_scores.append(score)
    return num.mean(silhouette_scores) if silhouette_scores else 0

def plot_silhouette(k_values, silhouette_scores):
    plot.figure(figsize=(15, 9))
    plot.plot(k_values, silhouette_scores, marker='D')
    plot.xlabel('no. clusters - s')
    plot.ylabel('silhouette coefficients')
    plot.title('plotting the ratios of silhouette and clusters for bisecting k-means')
    plot.grid(True)
    plot.savefig('BisectingKMeans.png')
    plot.show()
    


def main():
    data = load_data('dataset')
    silhouette_scores = []
    k_values = list(range(1, 10))
    for k in k_values:
        clusters, centroids = bisecting_kmeans(data, k)
        score = compute_silhouette(data, clusters, centroids)
        silhouette_scores.append(score)
    

    plot_silhouette(k_values, silhouette_scores)

main()
