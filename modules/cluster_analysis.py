import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from typing import Any, List, Optional, Tuple
import logging
import modules.globals

logger = logging.getLogger(__name__)

def find_cluster_centroids(embeddings, max_k=10) -> Any:
    """
    Identifies optimal face clusters using KMeans and silhouette scoring
    
    Args:
        embeddings: Face embedding vectors
        max_k: Maximum number of clusters to consider
        
    Returns:
        Array of optimal cluster centroids
    """
    try:
        if len(embeddings) < 2:
            logger.warning("Not enough embeddings for clustering analysis")
            return embeddings  # Return the single embedding as its own cluster
            
        # Use settings from globals if available
        max_k = getattr(modules.globals, 'max_cluster_k', max_k)
        kmeans_init = getattr(modules.globals, 'kmeans_init', 'k-means++')
        
        # Try silhouette method first
        best_k = 2  # Start with minimum viable cluster count
        best_score = -1
        best_centroids = None
        
        # We need at least 3 samples to calculate silhouette score
        if len(embeddings) >= 3:
            # Find optimal k using silhouette analysis
            for k in range(2, min(max_k+1, len(embeddings))):
                try:
                    kmeans = KMeans(n_clusters=k, init=kmeans_init, n_init=10, random_state=0)
                    labels = kmeans.fit_predict(embeddings)
                    
                    # Calculate silhouette score
                    score = silhouette_score(embeddings, labels)
                    
                    if score > best_score:
                        best_score = score
                        best_k = k
                        best_centroids = kmeans.cluster_centers_
                except Exception as e:
                    logger.warning(f"Error during silhouette analysis for k={k}: {str(e)}")
                    continue
        
        # Fallback to elbow method if silhouette failed or for small datasets
        if best_centroids is None:
            inertia = []
            cluster_centroids = []
            K = range(1, min(max_k+1, len(embeddings)+1))

            for k in K:
                kmeans = KMeans(n_clusters=k, init=kmeans_init, random_state=0)
                kmeans.fit(embeddings)
                inertia.append(kmeans.inertia_)
                cluster_centroids.append({"k": k, "centroids": kmeans.cluster_centers_})

            if len(inertia) > 1:
                diffs = [inertia[i] - inertia[i+1] for i in range(len(inertia)-1)]
                best_idx = diffs.index(max(diffs))
                best_centroids = cluster_centroids[best_idx + 1]['centroids']
            else:
                # Just one cluster
                best_centroids = cluster_centroids[0]['centroids']
        
        return best_centroids
        
    except Exception as e:
        logger.error(f"Error in cluster analysis: {str(e)}")
        # Return a single centroid (mean of all embeddings) as fallback
        return np.mean(embeddings, axis=0, keepdims=True)

def find_closest_centroid(centroids: list, normed_face_embedding) -> Optional[Tuple[int, np.ndarray]]:
    """
    Find the closest centroid to a face embedding
    
    Args:
        centroids: List of cluster centroids
        normed_face_embedding: Normalized face embedding vector
        
    Returns:
        Tuple of (centroid index, centroid vector) or None if matching fails
    """
    try:
        centroids = np.array(centroids)
        normed_face_embedding = np.array(normed_face_embedding)
        
        # Validate input shapes
        if len(centroids.shape) != 2 or len(normed_face_embedding.shape) != 1:
            logger.warning(f"Invalid shapes: centroids {centroids.shape}, embedding {normed_face_embedding.shape}")
            return None
            
        # Calculate similarity (dot product) between embedding and each centroid
        similarities = np.dot(centroids, normed_face_embedding)
        closest_centroid_index = np.argmax(similarities)
        
        return closest_centroid_index, centroids[closest_centroid_index]
    except Exception as e:
        logger.error(f"Error finding closest centroid: {str(e)}")
        return None