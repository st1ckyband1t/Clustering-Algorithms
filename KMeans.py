import numpy as num
import random
import matplotlib.pyplot as plot

#we've imported the libraries ive used

#the following method loads a file and reads the data
def load_data(fname):
    features = []
    with open(fname) as F:
        #first_line = next(F).strip().split(' ')
        #print("Number of columns:", len(first_line)) (was essentially using these instructions to debug and handle errors)
        for line in F:
            p = line.strip().split(' ') #(telling python how to split the data)
            features.append(num.array(p[1:], dtype=float))  #this instruction bassically appends the array "features" with data from column 2 to the end
    return num.array(features)

def ComputeDistance(a, b):
    return num.linalg.norm(a - b) #captures the euclidean distance

def initialSelection(x, k):
    random.seed(20) #setting the random seed to 20 to get the same results consistently
    indices = random.sample(range(x.shape[0]), k)
    return x[indices, :]

def clustername(x, k, maxIter=300):
    centroids = initialSelection(x, k)
    for i in range(maxIter):
        #this method basically assigns the clusters
        clusters = {i: [] for i in range(k)}
        for point in x:
            distances = [ComputeDistance(point, centroid) for centroid in centroids]
            minDistanceIndex = distances.index(min(distances))
            clusters[minDistanceIndex].append(point)
        
        #next we update the centroids
        new_centroids = num.array([num.mean(clusters[i], axis=0) if clusters[i] else centroids[i] for i in range(k)])
        if num.allclose(centroids, new_centroids, atol=1e-6):
            break
        centroids = new_centroids
    
    return clusters, centroids

def silhouette_score(x, clusters, centroids):
    silhouette_scores = []
    for i, cluster in clusters.items():
        if len(cluster) < 2:
            continue
        intra_distances = [ComputeDistance(point, centroids[i]) for point in cluster]#calling compute distances to calculate the distances for us
        a = num.mean(intra_distances)
        #print(a) (debugging uses)
        inter_cluster_distances = []
        for j, other_centroid in enumerate(centroids):#this loop works for each centroid in our graph
            if i != j and clusters[j]:
                inter_distances = [ComputeDistance(point, other_centroid) for point in cluster]
                inter_cluster_distances.append(num.mean(inter_distances))
        if inter_cluster_distances:
            b = min(inter_cluster_distances)
            #print (b)
            score = (b - a) / max(a, b)#this is the silhouette score
            silhouette_scores.append(score)
    return num.mean(silhouette_scores) if silhouette_scores else 0  #made sure to have a default value for the silhouette since its easier to troubleshoot then

#plotting the graph
def plot_silhouette(k_values, silhouette_scores):
    plot.figure(figsize=(15, 9))
    plot.plot(k_values, silhouette_scores, marker='D')
    plot.xlabel('no. clusters - K')
    plot.ylabel('silhouette coefficients')
    plot.title('plotting the ratios of silhouette and clusters')
    plot.grid(True)
    plot.savefig('Kmeans.png')
    plot.show()

#calling our methods from the main fucntion
def main():
    data = load_data('dataset')
    silhouette_scores = []
    k_values = list(range(1, 10))
    #the following loop clusters with k values ranging from 1 to 9
    for k in k_values:
        clusters, centroids = clustername(data, k, )
        score = silhouette_score(data, clusters, centroids)
        silhouette_scores.append(score)
    
    plot_silhouette(k_values, silhouette_scores)

main()
