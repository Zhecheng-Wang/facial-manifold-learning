import numpy as np
from collections import deque, defaultdict
from typing import List, Set, Tuple, Dict

def build_adjacency_list(vertices: np.ndarray, faces: np.ndarray) -> Dict[int, Set[int]]:
    """
    Build adjacency list from mesh vertices and faces.
    
    Args:
        vertices: Array of shape (N, 3) containing vertex coordinates
        faces: Array of shape (M, 3) containing face vertex indices
    
    Returns:
        Dictionary mapping vertex index to set of adjacent vertex indices
    """
    adjacency = defaultdict(set)
    
    for face in faces:
        # Each face connects three vertices
        v0, v1, v2 = face
        adjacency[v0].update([v1, v2])
        adjacency[v1].update([v0, v2])
        adjacency[v2].update([v0, v1])
    
    return adjacency

def compute_geodesic_distances(adjacency: Dict[int, Set[int]], 
                             start_vertices: List[int], 
                             max_distance: int = None) -> Dict[int, int]:
    """
    Compute geodesic distances (shortest path in terms of edge count) from start vertices.
    
    Args:
        adjacency: Adjacency list representation of the mesh
        start_vertices: List of starting vertex indices
        max_distance: Maximum distance to compute (None for unlimited)
    
    Returns:
        Dictionary mapping vertex index to minimum distance from any start vertex
    """
    distances = {}
    queue = deque()
    
    # Initialize with start vertices
    for start_v in start_vertices:
        distances[start_v] = 0
        queue.append((start_v, 0))
    
    while queue:
        current_vertex, current_dist = queue.popleft()
        
        # Skip if we've found a shorter path to this vertex
        if current_vertex in distances and distances[current_vertex] < current_dist:
            continue
            
        # Stop if we've reached maximum distance
        if max_distance is not None and current_dist >= max_distance:
            continue
        
        # Explore neighbors
        for neighbor in adjacency.get(current_vertex, []):
            new_dist = current_dist + 1
            
            if neighbor not in distances or distances[neighbor] > new_dist:
                distances[neighbor] = new_dist
                queue.append((neighbor, new_dist))
    
    return distances

def compute_vertex_assignments(vertices: np.ndarray, 
                             faces: np.ndarray,
                             key_point_set_A: List[int], 
                             key_point_set_B: List[int], 
                             K: int) -> Tuple[Set[int], Set[int]]:
    """
    Compute vertex assignments based on geodesic distances from key point sets.
    
    Args:
        vertices: Array of shape (N, 3) containing vertex coordinates
        faces: Array of shape (M, 3) containing face vertex indices
        key_point_set_A: List of vertex indices for key points A
        key_point_set_B: List of vertex indices for key points B
        K: Maximum edge distance for inclusion in set_A
    
    Returns:
        Tuple of (set_A, set_B) where:
        - set_A: vertices within K edges of key_point_set_A but not within K edges of key_point_set_B
        - set_B: remaining vertices
    """
    # Build adjacency list
    adjacency = build_adjacency_list(vertices, faces)
    
    # Compute distances from key point sets
    distances_A = compute_geodesic_distances(adjacency, key_point_set_A, K)
    distances_B = compute_geodesic_distances(adjacency, key_point_set_B, K)
    
    # Find vertices within K edges of key_point_set_A
    vertices_near_A = {v for v, dist in distances_A.items() if dist <= K}
    
    # Find vertices within K edges of key_point_set_B
    vertices_near_B = {v for v, dist in distances_B.items() if dist <= K}
    
    # set_A: vertices within K edges of A but NOT within K edges of B
    set_A = vertices_near_A - vertices_near_B
    
    # set_B: all remaining vertices
    all_vertices = set(range(len(vertices)))
    set_B = all_vertices - set_A
    
    return set_A, set_B

def visualize_assignments(vertices: np.ndarray,
                         faces: np.ndarray,
                         set_A: Set[int],
                         set_B: Set[int],
                         key_point_set_A: List[int],
                         key_point_set_B: List[int]) -> None:
    """
    Print statistics about the vertex assignments.
    
    Args:
        vertices: Array of shape (N, 3) containing vertex coordinates
        faces: Array of shape (M, 3) containing face vertex indices
        set_A: Vertices assigned to set A
        set_B: Vertices assigned to set B
        key_point_set_A: Original key points A
        key_point_set_B: Original key points B
    """
    total_vertices = len(vertices)
    
    print(f"Mesh Statistics:")
    print(f"  Total vertices: {total_vertices}")
    print(f"  Total faces: {len(faces)}")
    print(f"  Key points A: {len(key_point_set_A)}")
    print(f"  Key points B: {len(key_point_set_B)}")
    print()
    print(f"Assignment Results:")
    print(f"  Set A size: {len(set_A)} ({len(set_A)/total_vertices*100:.1f}%)")
    print(f"  Set B size: {len(set_B)} ({len(set_B)/total_vertices*100:.1f}%)")
    print(f"  Total assigned: {len(set_A) + len(set_B)}")

# Example usage
def example_usage():
    """
    Example demonstrating how to use the vertex assignment algorithm.
    """
    # Create a simple mesh (triangle strip)
    vertices = np.array([
        [0, 0, 0], [1, 0, 0], [2, 0, 0], [3, 0, 0],
        [0, 1, 0], [1, 1, 0], [2, 1, 0], [3, 1, 0],
        [0, 2, 0], [1, 2, 0], [2, 2, 0], [3, 2, 0]
    ])
    
    faces = np.array([
        [0, 1, 4], [1, 5, 4], [1, 2, 5], [2, 6, 5],
        [2, 3, 6], [3, 7, 6], [4, 5, 8], [5, 9, 8],
        [5, 6, 9], [6, 10, 9], [6, 7, 10], [7, 11, 10]
    ])
    
    # Define key point sets
    key_point_set_A = [0, 1]  # Bottom left corner
    key_point_set_B = [10, 11]  # Top right corner
    K = 2  # Maximum edge distance
    
    # Compute assignments
    set_A, set_B = compute_vertex_assignments(
        vertices, faces, key_point_set_A, key_point_set_B, K
    )
    
    # Display results
    visualize_assignments(vertices, faces, set_A, set_B, key_point_set_A, key_point_set_B)
    
    print(f"\nDetailed Results:")
    print(f"  Set A vertices: {sorted(list(set_A))}")
    print(f"  Set B vertices: {sorted(list(set_B))}")

if __name__ == "__main__":
    example_usage()