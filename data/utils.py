import trimesh
import numpy as np

import torch
from torch_cluster import knn_graph


def load_obj(filepath):
    """
    Load an OBJ file using trimesh.

    Paraneters:
        filepath (str): Path to the OBJ file.

    Returns:
        mesh (trimesh.Trimesh): The loaded mesh.
        vertices (np.ndarray): Array of vertex positions (N, 3).
        faces (np.ndarray): Array of face indices (M, 3).
        normals (np.ndarray): Array of vertex normals (N, 3).
    """
    mesh = trimesh.load(filepath, force='mesh')
    # If the file contains a scene with multiple geometries, take the first one.
    if not isinstance(mesh, trimesh.Trimesh):
        mesh = list(mesh.geometry.values())[0]
    vertices = mesh.vertices.astype(np.float32)
    faces = mesh.faces.astype(np.int64)
    normals = mesh.vertex_normals.astype(np.float32)
    return mesh, vertices, faces, normals

def parse_rig_info(filepath):
    """
    Parse a rig info txt file from the RigNet dataset.

    Expected file format (example):

        joints pelvis 0.00000000 0.43533900 -0.02258920
        joints L_hip 0.06730640 0.43533900 -0.02258920
        joints R_hip -0.06730640 0.43533800 -0.02258920
        ... (more joint lines)
        root pelvis
        skin 0 head 1.0000
        skin 1 head 1.0000
        ... (more skin lines)
        hier pelvis L_hip
        hier pelvis R_hip
        hier pelvis spine1
        ... (more hierarchy lines)

    This function does the following:
      - For lines starting with "joint" or "joints": it extracts the joint name and its 3D position.
      - For the line starting with "root": it records the root joint name.
      - For lines starting with "hier": it records parent-child pairs.
      - For lines starting with "skin": it builds a dictionary mapping vertex IDs to a dictionary
        of {joint_name: weight}. The vertex IDs are assumed to follow the vertex order in the OBJ file.

    Args:
        filepath (str): Path to the rig info text file.

    Returns:
        bone_features (np.ndarray): Array of joint positions of shape (B, 3) (ordered by joint names).
        root_joint (str): The name of the root joint.
        bone_hierarchy (list): A list of (parent, child) tuples.
        skin_weights_dict (dict): Mapping from vertex_id (int) to {joint_name: weight}.
        bone_names (list): Sorted list of joint names (keys from bone_features).
    """
    bone_features_dict = {}
    root_joint = None
    bone_hierarchy = []
    skin_weights_dict = {}

    with open(filepath, "r") as f:
        lines = f.readlines()

    for line in lines:
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        # Use startswith to handle both "joint" and "joints"
        token = parts[0].lower()
        if token.startswith("joint"):
            # Expected format: "joints joint_name X Y Z"
            if len(parts) < 5:
                continue  # Skip malformed lines.
            joint_name = parts[1]
            pos = list(map(float, parts[2:5]))
            bone_features_dict[joint_name] = pos
        elif token == "root":
            # Expected format: "root joint_name"
            if len(parts) >= 2:
                root_joint = parts[1]
        elif token == "hier":
            # Expected format: "hier parent_joint child_joint"
            if len(parts) >= 3:
                bone_hierarchy.append((parts[1], parts[2]))
        elif token == "skin":
            # Expected format: "skin vertex_id joint_name1 weight1 joint_name2 weight2 ..."
            if len(parts) < 3:
                continue
            vertex_id = int(parts[1])
            weights = {}
            for i in range(2, len(parts), 2):
                if i + 1 < len(parts):
                    joint = parts[i]
                    weight = float(parts[i+1])
                    weights[joint] = weight
            skin_weights_dict[vertex_id] = weights

    # Create a sorted list of bone names for consistent ordering.
    bone_names = sorted(bone_features_dict.keys())
    bone_features = np.array([bone_features_dict[name] for name in bone_names], dtype=np.float32)

    return bone_features, root_joint, bone_hierarchy, skin_weights_dict, bone_names

def convert_skin_weights(skin_weights_dict, num_vertices, bone_names):
    """
    Convert per-vertex skin weight dictionaries into an array of shape (N, B).

    If the maximum vertex index in skin_weights_dict equals num_vertices,
    we assume the rig file is 1-indexed and subtract 1 from every key.

    Args:
        skin_weights_dict (dict): Mapping from vertex_id to {joint_name: weight}.
        num_vertices (int): Total number of vertices (N).
        bone_names (list): List of bone names (length B).

    Returns:
        weights_tensor (np.ndarray): Array of shape (N, B) with normalized skin weights.
    """
    N = num_vertices
    B = len(bone_names)

    # Check if the maximum vertex id equals N. If so, assume 1-indexing.
    if skin_weights_dict:
        max_vid = max(skin_weights_dict.keys())
        if max_vid == N:
            # Convert to 0-indexing.
            skin_weights_dict = {vid - 1: weight_dict for vid, weight_dict in skin_weights_dict.items()}

    weights_tensor = np.zeros((N, B), dtype=np.float32)
    for vid, weight_dict in skin_weights_dict.items():
        # (Assuming now that vid is in [0, N-1])
        for joint, weight in weight_dict.items():
            if joint in bone_names:
                j = bone_names.index(joint)
                weights_tensor[vid, j] = weight
    # Normalize each row so that the sum equals 1.
    row_sums = weights_tensor.sum(axis=1, keepdims=True) + 1e-8
    weights_tensor = weights_tensor / row_sums
    return weights_tensor

def compute_vertex_neighbors_knn(vertices, k):
    """
    Compute vertex neighbors using torch_cluster.knn_graph.

    This function leverages a highly optimized routine from torch_cluster to compute
    neighbor indices based on Euclidean distance. It is more efficient and robust compared
    to manually computing neighbors from face connectivity.

    Paraneters:
        vertices (torch.Tensor): (N, 3) tensor.
        k (int): Number of neighbors.

    Returns:
        neighbor_indices (torch.LongTensor): (N, k) tensor of neighbor indices.
    """
    edge_index = knn_graph(vertices, k=k, loop=True)  # (2, E)
    N = vertices.size(0)
    if edge_index.size(1) == N * k:
        neighbor_indices = edge_index[1].view(N, k)
    else:
        neighbor_indices = torch.zeros(N, k, dtype=torch.long, device=vertices.device)
        for i in range(N):
            idx = (edge_index[0] == i).nonzero(as_tuple=False).squeeze()
            if idx.numel() < k:
                nbrs = edge_index[1][idx]
                pad_size = k - nbrs.numel()
                nbrs = torch.cat([nbrs, torch.full((pad_size,), i, dtype=torch.long, device=vertices.device)])
            else:
                nbrs = edge_index[1][idx][:k]
            neighbor_indices[i] = nbrs
    return neighbor_indices

def compute_vertex_adj_from_faces(num_vertices, faces):
    """
    Compute a normalized vertex adjacency matrix from face connectivity.

    Each vertex is connected to every other vertex that shares a face with it.

    Parameters:
        num_vertices (int): Total number of vertices.
        faces (np.ndarray): Array of face indices (M, 3).

    Returns:
        adj (torch.FloatTensor): (num_vertices, num_vertices) normalized adjacency matrix.
    """
    adj = np.zeros((num_vertices, num_vertices), dtype=np.float32)
    for face in faces:
        for i in face:
            for j in face:
                if i != j:
                    adj[i, j] = 1.0
    adj += np.eye(num_vertices, dtype=np.float32)
    row_sum = adj.sum(axis=1, keepdims=True) + 1e-8
    adj = adj / row_sum
    return torch.tensor(adj, dtype=torch.float32)

def compute_rich_bone_features(bone_features, bone_hierarchy, bone_names):
    """
    Compute additional features for each bone given the original joint positions.

    For each bone (joint), if it has a parent (according to bone_hierarchy),
    compute:
      - bone length: the Euclidean distance between the joint and its parent.
      - relative direction: the unit vector from the parent's position to the joint's position.
    If no parent is found (e.g. for the root), these additional features are set to zeros.

    Args:
        bone_features (np.ndarray): (B, 3) array of joint positions.
        bone_hierarchy (list): List of (parent, child) tuples.
        bone_names (list): Sorted list of joint names corresponding to the rows of bone_features.

    Returns:
        rich_features (np.ndarray): Array of shape (B, 7) where each row is:
            [x, y, z, bone_length, dx, dy, dz]
    """
    B = len(bone_names)
    # Create a dictionary mapping bone name to its 3D position.
    bone_pos_dict = {name: bone_features[i] for i, name in enumerate(bone_names)}
    # Create a mapping from child joint to its parent based on the hierarchy.
    parent_map = {}
    for parent, child in bone_hierarchy:
        if child not in parent_map:  # Use first occurrence if multiple.
            parent_map[child] = parent

    rich_features = []
    for name in bone_names:
        pos = bone_pos_dict[name]  # (3,)
        if name in parent_map:
            parent_name = parent_map[name]
            parent_pos = bone_pos_dict.get(parent_name, None)
            if parent_pos is not None:
                diff = pos - parent_pos
                length = np.linalg.norm(diff)
                if length > 1e-6:
                    direction = diff / length
                else:
                    direction = np.zeros_like(diff)
            else:
                length = 0.0
                direction = np.zeros(3)
        else:
            # For the root or if no parent is defined, set extra features to zero.
            length = 0.0
            direction = np.zeros(3)
        # Concatenate original position (3), bone length (1) and relative direction (3).
        rich_feat = np.concatenate([pos, [length], direction], axis=0)  # (7,)
        rich_features.append(rich_feat)
    rich_features = np.stack(rich_features, axis=0)  # (B, 7)
    return rich_features

def compute_geodesic_knn_from_matrix(surface_geodesic, k):
    """
    Given a full (N, N) surface geodesic distance matrix, compute the indices of the k-nearest neighbors for each vertex based on geodesic distance.

    Parameters:
        surface_geodesic (np.ndarray): Dense geodesic matrix of shape (N, N).
        k (int): Number of neighbors.

    Returns:
        knn_indices (torch.LongTensor): Tensor of shape (N, k) with neighbor indices.
    """
    # Exclude self by setting the diagonal to infinity.
    d = surface_geodesic.copy()
    np.fill_diagonal(d, np.inf)
    knn_indices = np.argsort(d, axis=1)[:, :k]
    return torch.tensor(knn_indices, dtype=torch.long)

def build_geodesic_graph_from_knn(surface_geodesic, knn_indices):
    """
    Build a geodesic-based graph (edge_index and edge weights) from the knn indices.

    Parameters:
        surface_geodesic (np.ndarray): Dense geodesic matrix of shape (N, N).
        knn_indices (torch.LongTensor): Tensor of shape (N, k) with neighbor indices.

    Returns:
        edge_index (Tensor): shape (2, N*k)
        edge_weight (Tensor): shape (N*k, 1) with the geodesic distance for each edge.
    """
    N = surface_geodesic.shape[0]
    k = knn_indices.shape[1]
    src = torch.arange(N).unsqueeze(1).repeat(1, k).view(-1)
    dst = knn_indices.view(-1)
    edge_index = torch.stack([src, dst], dim=0)
    # Gather corresponding geodesic distances.
    edge_weight = []
    for i in range(N):
        for j in knn_indices[i]:
            edge_weight.append(surface_geodesic[i, j.item()])
    edge_weight = torch.tensor(edge_weight, dtype=torch.float32)
    return edge_index, edge_weight
