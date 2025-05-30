import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from typing import Any, List, Tuple


def find_cluster_centroids(embeddings: List[Any], max_k: int = 10) -> Any:
    """Find optimal cluster centroids for a set of embeddings using KMeans."""
    inertia = []
    cluster_centroids = []
    K = range(1, max_k+1)

    for k in K:
        try:
            kmeans = KMeans(n_clusters=k, random_state=0)
            kmeans.fit(embeddings)
            inertia.append(kmeans.inertia_)
            cluster_centroids.append({"k": k, "centroids": kmeans.cluster_centers_})
        except Exception as e:
            print(f"KMeans failed for k={k}: {e}")

    if len(inertia) < 2:
        return cluster_centroids[0]['centroids'] if cluster_centroids else []

    diffs = [inertia[i] - inertia[i+1] for i in range(len(inertia)-1)]
    optimal_centroids = cluster_centroids[diffs.index(max(diffs)) + 1]['centroids']

    return optimal_centroids


def find_closest_centroid(centroids: List[Any], normed_face_embedding: Any) -> Tuple[int, Any]:
    """Find the index and value of the centroid closest to the given embedding."""
    try:
        centroids = np.array(centroids)
        normed_face_embedding = np.array(normed_face_embedding)
        similarities = np.dot(centroids, normed_face_embedding)
        closest_centroid_index = np.argmax(similarities)

        return closest_centroid_index, centroids[closest_centroid_index]
    except Exception as e:
        print(f"Error in find_closest_centroid: {e}")
        return -1, None