import numpy as np
import open3d as o3d
from scipy.sparse import lil_matrix
from scipy.sparse.csgraph import dijkstra

def compute_surface_geodesic_features(mesh_path):
    """
    Calculate the geodesic distances between points on the surface of a 3D mesh.
    From:
        https://github.com/zhan-xu/RigNet/blob/master/geometric_proc/compute_surface_geodesic.py
        https://github.com/zhan-xu/RigNet/blob/master/geometric_proc/common_ops.py
    Parameters:
    mesh_path (string): The path to the input 3D mesh.

    Returns:
    np.ndarray: A 2D array where element (i, j) represents the geodesic distance between vertex i and vertex j.
    """
    mesh = o3d.io.read_triangle_mesh(mesh_path)
    # Densely sample 4000 points (proccessed meshes in the Rignet dataset are between 1000 and 5000 vertices)
    # from the mesh surface using Poisson disk sampling.
    # sample_points_poisson_disk generates uniformly distributed points across the mesh surface
    # while maintaining minimum distances between points. This ensures better coverage and
    # more stable geodesic calculations compared to random sampling.
    samples = mesh.sample_points_poisson_disk(number_of_points=4000)
    pts = np.asarray(samples.points)  # Shape: (4000, 3)
    pts_normal = np.asarray(samples.normals)  # Shape: (4000, 3)

    # Try to get normals from the sampling.
    pts_normal = np.asarray(samples.normals)
    # If normals are not returned, fallback to mesh.vertex_normals.
    if pts_normal.size == 0:
        print("Warning: sampled normals are empty; falling back to mesh.vertex_normals.")
        pts_normal = np.asarray(mesh.vertex_normals)
        # If still empty, raise an error.
        if pts_normal.size == 0:
            raise ValueError("No normals available for this mesh.")

    # Number of sampled points
    N = len(pts)  # N = 4000

    # Compute the Euclidean distance between each pair of sampled points
    # verts_dist[i, j] contains the distance between pts[i] and pts[j]
    verts_dist = np.sqrt(np.sum((pts[np.newaxis, ...] - pts[:, np.newaxis, :]) ** 2, axis=2))  # Shape: (N, N)

    # np.newaxis allows broadcasting for operations like:
    # pts[np.newaxis, ...] - pts[:, np.newaxis, :]
    # Shape: (2, 2, 3) = (1, 2, 3) broadcasted with (2, 1, 3)
    # Result: pairwise differences between all points
    # Example:
    # pts = np.array([[1,2,3], [4,5,6]])  # Shape: (2, 3)
    # pts[np.newaxis, ...])    # Shape: (1, 2, 3) [[[1, 2, 3], [4, 5, 6]]]
    # pts[:, np.newaxis, :]   # Shape: (2, 1, 3) [[[1, 2, 3] ], [[4, 5, 6]]]
    # pts = np.array([
    #     [1, 0, 0],  # point 1
    #     [0, 1, 0]   # point 2
    # ])
    # 1. pts[np.newaxis, ...] creates shape (1, 2, 3):
    # [[[1, 0, 0],
    #   [0, 1, 0]]]
    # 2. pts[:, np.newaxis, :] creates shape (2, 1, 3):
    # [[[1, 0, 0]],
    #  [[0, 1, 0]]]
    # 3. The subtraction broadcasts to shape (2, 2, 3):
    # [[[0, 0, 0],     # p1-p1: [1,0,0]-[1,0,0]
    #   [1,-1, 0]],    # p1-p2: [1,0,0]-[0,1,0]
    #  [[-1, 1, 0],    # p2-p1: [0,1,0]-[1,0,0]
    #   [0, 0, 0]]]    # p2-p2: [0,1,0]-[0,1,0]
    # 4. ** 2 squares each element
    # 5. np.sum(..., axis=2) sums along the last axis (coordinates)
    # Result shape (2, 2):
    # [[0,   2],    # distances from point 1 to all points
    #  [2,   0]]    # distances from point 2 to all points

    # Determine the indices that would sort each row of verts_dist
    # verts_nn[i] contains the indices of pts sorted by distance to pts[i]
    verts_nn = np.argsort(verts_dist, axis=1)  # Shape: (N, N)

    # Initialize a sparse matrix to represent the graph's adjacency matrix
    conn_matrix = lil_matrix((N, N), dtype=np.float32)  # Shape: (N, N)

    # Construct the adjacency matrix based on nearest neighbors and normal similarity
    for p in range(N):
        # Select the 5 nearest neighbors to point p (excluding itself)
        nn_p = verts_nn[p, 1:6]  # Shape: (5,)

        # Compute the norms of the normals for point p and its neighbors
        norm_nn_p = np.linalg.norm(pts_normal[nn_p], axis=1)  # Shape: (5,)
        norm_p = np.linalg.norm(pts_normal[p])  # Scalar

        # Calculate the cosine similarity between the normal of point p and its neighbors
        cos_similar = np.dot(pts_normal[nn_p], pts_normal[p]) / (norm_nn_p * norm_p + 1e-10)  # Shape: (5,)

        # Filter neighbors based on cosine similarity threshold
        nn_p = nn_p[cos_similar > -0.5]  # Shape: (<=5,)

        # Set the distances to the valid neighbors in the adjacency matrix
        conn_matrix[p, nn_p] = verts_dist[p, nn_p]

    # This combination of the usage of the metrics of Euclideian and Cosine similarity helps to:
    # - Find nearby points (Euclidean)
    # - Ensure they belong to the same surface region by checking normal alignment (Cosine)
    # This prevents connections across thin features or separate parts of the mesh that are close
    # in 3D space but not actually connected on the surface.
    # For example, two points might be close in space but on opposite sides of a thin wall
    # (the normal comparison would prevent them from being connected.

    # Compute the shortest paths (geodesic distances) between all pairs of nodes
    dist, predecessors = dijkstra(
        conn_matrix, directed=False, indices=range(N), return_predecessors=True, unweighted=False
    )  # dist Shape: (N, N), predecessors Shape: (N, N)

    # Identify positions in the distance matrix with infinite values (unreachable nodes)
    inf_pos = np.argwhere(np.isinf(dist))  # Shape: (num_infinite_entries, 2)

    if len(inf_pos) > 0:
        # Compute the Euclidean distances between all pairs of sampled points
        euc_distance = np.sqrt(np.sum((pts[np.newaxis, ...] - pts[:, np.newaxis, :]) ** 2, axis=2))  # Shape: (N, N)

        # Replace infinite distances with Euclidean distance + 8.0
        dist[inf_pos[:, 0], inf_pos[:, 1]] = 8.0 + euc_distance[inf_pos[:, 0], inf_pos[:, 1]]

    # Retrieve the original mesh vertices
    verts = np.array(mesh.vertices)  # Shape: (num_vertices, 3)

    # Compute the Euclidean distance between each original vertex and each sampled point
    vert_pts_distance = np.sqrt(np.sum((verts[np.newaxis, ...] - pts[:, np.newaxis, :]) ** 2, axis=2))  # Shape: (N, num_vertices)

    # For each original vertex, find the index of the nearest sampled point
    vert_pts_nn = np.argmin(vert_pts_distance, axis=0)  # Shape: (num_vertices,)

    # Construct the geodesic distance matrix for the original mesh vertices
    surface_geodesic = dist[vert_pts_nn, :][:, vert_pts_nn]  # Shape: (num_vertices, num_vertices)

    return surface_geodesic