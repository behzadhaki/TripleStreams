import numpy as np
from scipy.spatial.distance import cdist

groove      = np.array([1, 0, 0, 0])
response_a  = np.array([0, 0, 1, 0])
response_b  = np.array([0, 1, 1, 0])

# calculate cosine distance between response_a/b and groove
def cosine_distance(a, b):
    dot_product = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    return 1 - (dot_product / (norm_a * norm_b))

def euclidean_distance(a, b):
    return np.linalg.norm(a - b)

def Jaccard_similarity(a, b):
    intersection = np.sum(np.minimum(a, b))
    union = np.sum(np.maximum(a, b))
    return (intersection / union)

def hamming_distance(a, b):
    if len(a) != len(b):
        raise ValueError("Sequences must be of equal length")
    return sum(x != y for x, y in zip(a, b))/ len(a)



print("Cosine Distance:\n-----------------\nA/Groove\t\tB/Groove")
print(f"{cosine_distance(response_a, groove):.4f}\t\t\t{cosine_distance(response_b, groove):.4f}")

print("\nEuclidean Distance:\n-------------------\nA/Groove\t\tB/Groove")
print(f"{euclidean_distance(response_a, groove):.4f}\t\t\t{euclidean_distance(response_b, groove):.4f}")

print("\nJaccard Similarity:\n---------------\nA/Groove\t\tB/Groove")
print(f"{Jaccard_similarity(response_a, groove):.4f}\t\t\t{Jaccard_similarity(response_b, groove):.4f}")

print("\nhamming_distance:\n----------------\nA/Groove\t\tB/Groove")
print(f"{hamming_distance(response_a, groove):.4f}\t\t\t{hamming_distance(response_b, groove):.4f}")
