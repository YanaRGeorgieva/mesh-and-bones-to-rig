import os
import numpy as np

import torch
from torch_geometric.data import Data, Dataset

from preprocess_utils.surface_geo import compute_surface_geodesic_features
from preprocess_utils.volumetric_geo import compute_volumetric_geodesic
from data.utils import build_geodesic_graph_from_knn, compute_geodesic_knn_from_matrix, compute_rich_bone_features, compute_vertex_adj_from_faces, convert_skin_weights, load_obj, parse_rig_info

class MeshBonesToRigDataset(Dataset):
    """
    MeshBonesToRigDataset for the pre-processed data stored in a folder structure used to train originally RigNet.
    https://github.com/zhan-xu/RigNet?tab=readme-ov-file#:~:text=Lib%5Csite%2Dpackages)-,Data,the%20root_folder%20to%20the%20directory%20you%20uncompress%20the%20pre%2Dprocessed%20data.,-Training

    Expected folder structure:
      - obj_remesh: Contains OBJ files.
      - rig_info_remesh: Contains corresponding rig info txt files.
      - (Other folders may be present, but we use these two for our rigging network.)

    Each model is assumed to have files with the same base name in these folders.

    This dataset loader uses trimesh to load the mesh and extract vertices, faces, and vertex normals.
    It parses the rig info txt file to extract:
      - Joint definitions (bone features).
      - Hierarchy information to build a bone adjacency matrix.
      - Skinning weights (target skin weights).

    Additionally, it computes vertex neighbor indices building a kNN graph and the vertex
    adjacency matrix from face connectivity.
    """
    def __init__(self, root_dir, cache_dir=None, k=8, allowed_names=None, transform=None):

        """
        Paraneters:
            root_dir (str): Directory containing model subdirectories.
            k (int): Number of nearest neighbors for each vertex
                (8 is chosen as a balanced, commonly effective default that captures local structure).
            allowed_names (list or set, optional): A collection of model base names to keep.
                If provided, only models with a base name in this collection will be included.
            transform (callable, optional): A function to apply to each sample.
        """
        self.root_dir = root_dir
        # Directories for each type of data.
        self.obj_dir = os.path.join(root_dir, "obj_remesh")
        self.rig_dir = os.path.join(root_dir, "rig_info_remesh")
        if cache_dir is None:
            self.cache_dir = os.path.join(root_dir, "precomputed")
        else:
            self.cache_dir = cache_dir
        os.makedirs(self.cache_dir, exist_ok=True)

        # List all OBJ files.
        all_files = [f for f in os.listdir(self.obj_dir) if f.lower().endswith(".obj")]

        if allowed_names is not None:
            # Filter: only keep files whose base name (without extension) is in allowed_names.
            self.obj_files = [f for f in all_files if os.path.splitext(f)[0] in allowed_names]
        else:
            self.obj_files = all_files

        self.k = k
        self.transform = transform

    def __len__(self):
        return len(self.obj_files)

    def __getitem__(self, idx):
        # Use the file at self.obj_files[idx]
        base_name = os.path.splitext(self.obj_files[idx])[0]
        # Paths to the files.
        obj_path = os.path.join(self.obj_dir, base_name + ".obj")
        rig_path = os.path.join(self.rig_dir, base_name + ".txt")

        # Load the mesh.
        mesh, vertices_np, faces_np, normals_np = load_obj(obj_path)
        N_obj = vertices_np.shape[0]

        # Parse rig info.
        bone_positions_np, root_joint, bone_hierarchy, skin_weights_dict, bone_names = parse_rig_info(rig_path)
        # Determine the "expected" number of vertices from the rig data.
        # For instance, if skin_weights_dict uses vertex IDs, get the maximum vertex id.
        if skin_weights_dict:
            expected_vertices = max(skin_weights_dict.keys()) + 1  # assuming 0-indexing
        else:
            expected_vertices = N_obj  # if no skin data is available, accept the OBJ count

        # Check for mismatch. There are some models that have different number of vertices in the rig info and the obj file.
        if N_obj != expected_vertices:
            # Skip this sample by trying the next index (wrap around if needed).
            print(f"Skipping sample {base_name}: OBJ has {N_obj} vertices, but rig expects {expected_vertices}")
            # Be cautious: if many samples are mismatched, this could loop indefinitely.
            # Alternatively, you could return None and then filter out None in your collate function.
            return self.__getitem__((idx + 1) % len(self))

        # Continue with processing:
        # Compute vertex neighbors using kNN.
        # vertices_tensor = torch.tensor(vertices_np, dtype=torch.float32)
        # vertex_neighbors = compute_vertex_neighbors_knn(vertices_tensor, self.k)  # (N_obj, k)

        # Compute vertex adjacency matrix from faces.
        vertex_adj = compute_vertex_adj_from_faces(N_obj, faces_np)  # (N_obj, N)

        # Convert skin weights to a numpy array (N_obj, B), where B = number of bones.
        target_skin_weights_np = convert_skin_weights(skin_weights_dict, N_obj, bone_names)

        # Ensure that bone_positions_np is 2-dimensional.
        # This can happen if, e.g., the rig info file contains only one joint.
        if bone_positions_np.ndim == 1:
            # Reshape to (1, -1) so that shape[1] is available.
            bone_positions_np = bone_positions_np.reshape(1, -1)

        # For bone features, we have (B, 3).
        B = len(bone_names)
        rich_bone_features = compute_rich_bone_features(bone_positions_np, bone_hierarchy, bone_names)  # (B, 7)

        # Optionally, pad to 8 dimensions (if your network expects 8):
        if rich_bone_features.shape[1] < 8:
            pad = np.zeros((rich_bone_features.shape[0], 8 - rich_bone_features.shape[1]), dtype=np.float32)
            rich_bone_features = np.concatenate([rich_bone_features, pad], axis=1)  # Now (B, 8)

        # Build bone adjacency matrix from hierarchy.
        bone_adj = np.zeros((B, B), dtype=np.float32)
        # Use the "hier" information: for each (parent, child) pair, set an edge.
        for parent, child in bone_hierarchy:
            if parent in bone_names and child in bone_names:
                i = bone_names.index(parent)
                j = bone_names.index(child)
                bone_adj[i, j] = 1
                bone_adj[j, i] = 1  # Assuming undirected for simplicity.
        # Add self-connections and normalize.
        bone_adj += np.eye(B, dtype=np.float32)
        row_sum = bone_adj.sum(axis=1, keepdims=True) + 1e-8
        bone_adj_norm = bone_adj / row_sum

        # Load or compute cached geometric features.
        vol_geo_path = os.path.join(self.cache_dir, base_name + "_volgeo.npy")
        surf_geo_path = os.path.join(self.cache_dir, base_name + "_surfgeo.npy")
        if os.path.exists(vol_geo_path) and os.path.exists(surf_geo_path):
            vol_geo_np = np.load(vol_geo_path).astype(np.float32)
            surf_geo_np = np.load(surf_geo_path).astype(np.float32)
        else:
            # Get bone positions from rig info.
            vol_geo_np = compute_volumetric_geodesic(vertices_np, faces_np, bone_positions_np) # (N_obj, B)
            surf_geo_np = compute_surface_geodesic_features(obj_path) # (N_obj, N_obj)

            vol_geo_path = os.path.join(self.cache_dir, base_name + "_volgeo.npy")
            surf_geo_path = os.path.join(self.cache_dir, base_name + "_surfgeo.npy")
            np.save(vol_geo_path, vol_geo_np)
            np.save(surf_geo_path, surf_geo_np)

        # Derive the k-NN indices from the full geodesic matrix.
        vertex_neighbors = compute_geodesic_knn_from_matrix(surf_geo_np, self.k) # (N_obj, k)

        # Here, we will let the network compute that from vertex_neighbors if desired.
        # Build geodesic k-NN graph (edge_index, edge_weight) from the full surface geodesic matrix.
        edge_index_tensor, geo_edge_weight_tensor = build_geodesic_graph_from_knn(surf_geo_np, vertex_neighbors)


        # Convert everything to torch tensors.
        vertices_tensor = torch.tensor(vertices_np, dtype=torch.float32)
        normals_tensor = torch.tensor(normals_np, dtype=torch.float32)
        target_skin_weights_tensor = torch.tensor(target_skin_weights_np, dtype=torch.float32)
        bone_positions_tensor = torch.tensor(bone_positions_np, dtype=torch.float32)
        bone_features_tensor = torch.tensor(rich_bone_features, dtype=torch.float32)
        bone_adj_tensor = torch.tensor(bone_adj_norm, dtype=torch.float32)
        vol_geo_tensor = torch.tensor(vol_geo_np, dtype=torch.float32)
        surface_geodesic_tensor =torch.tensor(surf_geo_np, dtype=torch.float32)

        data = Data(
            vertices=vertices_tensor,                               # (N_obj, 3)
            edge_index_geodesic=edge_index_tensor,                  # (2, N_obj*k)
            edge_attr_geodesic=geo_edge_weight_tensor.unsqueeze(1), # Set edge_attr to be the geodesic distance (reshaped to (N*k, 1)).
            vertex_neighbors=vertex_neighbors,                      # (N_obj, k) based on geodesic distances
            vertex_adj=vertex_adj,                                  # (N_obj, N_obj)
            vertex_normals=normals_tensor,                          # (N_obj, 3)
            bone_positions=bone_positions_tensor,                   # (B, 3)
            bone_features=bone_features_tensor,                     # (B, 8)
            bone_adj=bone_adj_tensor,                               # (B, B)
            target_skin_weights=target_skin_weights_tensor,         # (N_obj, B)
            volumetric_geodesic=vol_geo_tensor,                     # (N_obj, B)
            surface_geodesic=surface_geodesic_tensor                # (N_obj, N_obj)
        )

        # Add the number of vertices to the data.
        data.num_nodes = data.vertices.size(0)
        if self.transform:
            data = self.transform(data)

        return data