"""
Estudiante: Allan Barrantes
Carnet: B80986
"""
import numpy as np
import matplotlib.pyplot as plt
import random
from PIL import Image

def load_image(filename, resize=None):
    img = Image.open(filename)
    if resize != None:    
        img = img.resize(resize, Image.ANTIALIAS)
    data = np.array(img)
    return data
    
def euclidean_distance(p1,p2):
    return np.linalg.norm(p1 - p2)

def manhattan_distance(p1,p2):
    return sum(abs(val1-val2) for val1, val2 in zip(p1,p2))

def nearest_centroid(point, centroids, distance="euclidean"):
    dist = []
    for idx_centroid in range(len(centroids)):
        if distance == "euclidean":
            dist.append(euclidean_distance(point, centroids[idx_centroid]))
        else:
            dist.append(manhattan_distance(point, centroids[idx_centroid]))
    nearest = min(dist)
    return dist.index(nearest), nearest

def initialize_centroids(data, k, distance):
    # Init first centroid
    centroids = np.empty((k,3))
    centroid_idx_0 = random.randint(0, data.shape[0]-1)
    centroid_idx_1 = random.randint(0, data.shape[1]-1)
    centroids[0] = data[centroid_idx_0][centroid_idx_1]
    
    # Select the other centroids (k-means++ initialization)
    for c_idx in range(k - 1):
        dist = 0
        idx_0 = 0
        idx_1 = 0
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                if data[i,j] not in centroids:
                    for c in range(len(centroids)):
                        if distance == "euclidean":
                            temp_dist = euclidean_distance(data[i][j], centroids[c])
                        else:
                            temp_dist = manhattan_distance(data[i][j], centroids[c])
                        if temp_dist > dist:
                            dist = temp_dist
                            idx_0 = i
                            idx_1 = j
                        
        centroids[c_idx+1] = data[idx_0][idx_1]
    return centroids

def lloyd(data, k, iters=5, _type="means", distance="euclidean"):
    centroids = initialize_centroids(data, k, distance)
    clusters = {}
    for it in range(iters):
        for i in range(k):
            clusters[i] = []
        clusters_costs = [0] * k
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                c_idx, nearest_d = nearest_centroid(data[i, j], centroids, distance)
                clusters[c_idx].append(list(data[i,j]))
                clusters_costs[c_idx] += nearest_d
        # Recalculate centroids
        for c in range(k):
            if _type == "means":
                new_centr = [float(sum(col))/len(col) for col in zip(*clusters[c])]
            else: 
                # medoids
                medoid_cost = 0
                medoid_idx = random.randint(0, len(clusters[c])-1)
                for point in clusters[c]:
                    if distance == "euclidean":
                        medoid_cost += euclidean_distance(np.array(clusters[c][medoid_idx]), np.array(point))
                    else:
                        medoid_cost += manhattan_distance(clusters[c][medoid_idx], point)
                if medoid_cost < clusters_costs[c]:
                    new_centr = clusters[c][medoid_idx]
                else:
                    new_centr = centroids[c]
            centroids[c] = new_centr  
    return clusters

main_colors = 5
img_resize = (128,128)
points_resize = (128,128,3)
img = load_image('imgs/fight-club.webp', img_resize)
clusters = lloyd(img, main_colors, 10, "means", "euclidean")

# Get colour palette from the clusters
colour_palette = []
for i in range(main_colors):
    for j in range(len(clusters[i])):
        colour_palette.append(clusters[i][j])
colour_palette = np.array(colour_palette).reshape(points_resize)
plt.imshow(colour_palette)
plt.savefig('colour_palettes/fight-club_meanseuclidean.jpeg') 