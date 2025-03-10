import numpy as np
from pxr import Usd, UsdGeom, Gf, Sdf
import os
import open3d as o3d
from scipy.ndimage import gaussian_filter
from scipy.ndimage import label

# ===========================================
# Functions to process bounding boxes
# ===========================================


def compute_bbox(prim: Usd.Prim) -> Gf.Range3d:
    """
    Compute Bounding Box using ComputeWorldBound at UsdGeom.Imageable

    Args:
        prim: A prim to compute the bounding box.
    Returns: 
        A range (i.e. bounding box)
    """
    imageable: UsdGeom.Imageable = UsdGeom.Imageable(prim)
    time = Usd.TimeCode.Default()
    bound = imageable.ComputeWorldBound(time, UsdGeom.Tokens.default_)
    bound_range = bound.ComputeAlignedBox()
    return bound_range

def merge_bbox(prims):
    min_points = np.array([float('inf'), float('inf'), float('inf')])
    max_points = np.array([float('-inf'), float('-inf'), float('-inf')])
    for prim in prims:
        bbox = compute_bbox(prim)
        min_points = np.minimum(min_points, bbox.min)
        max_points = np.maximum(max_points, bbox.max)

    min_point = Gf.Vec3d(*min_points)
    max_point = Gf.Vec3d(*max_points)

    return Gf.Range3d(min_point, max_point)

def compute_bbox_volume(bbox):
    size = bbox.max - bbox.min
    volume = size[0] * size[1] * size[2]
    return volume

def calculate_bbox_iou(bbox_a, bbox_b):

    a_min, a_max = bbox_a.min, bbox_a.max
    b_min, b_max = bbox_b.min, bbox_b.max
    overlap_min = np.maximum(a_min, b_min)
    overlap_max = np.minimum(a_max, b_max)
    # print(overlap_min, overlap_max)
    overlap_size = overlap_max - overlap_min
    if any(size <= 0 for size in overlap_size):
        return False, 0.0
    intersection_volume = overlap_size[0] * overlap_size[1] * overlap_size[2]
    volume_a = (a_max[0] - a_min[0]) * (a_max[1] - a_min[1]) * (a_max[2] - a_min[2])
    volume_b = (b_max[0] - b_min[0]) * (b_max[1] - b_min[1]) * (b_max[2] - b_min[2])
    union_volume = volume_a + volume_b - intersection_volume
    iou = intersection_volume / union_volume

    return True, iou

def is_bbox_nearby(bbox_a, bbox_b, scale_factor=0.001):

    a_min, a_max = bbox_a.min, bbox_a.max
    b_min, b_max = bbox_b.min, bbox_b.max
    a_size = np.linalg.norm(a_max - a_min)
    b_size = np.linalg.norm(b_max - b_min)
    max_bbox_size = min(a_size, b_size)
    distance_threshold = max_bbox_size * scale_factor
    distance = 0.0
    for i in range(3):  
        if a_max[i] < b_min[i]:  
            distance += (b_min[i] - a_max[i]) ** 2
        elif b_max[i] < a_min[i]:  
            distance += (a_min[i] - b_max[i]) ** 2

    distance = np.sqrt(distance) 
    is_near = distance <= distance_threshold
    return is_near, distance

# ===========================================
# Functions to process the whole stage
# ===========================================
def find_stage_all_lights(stage):
    all_prims = stage.TraverseAll()
    light_num = 0
    for prim in all_prims:
        if prim.GetTypeName() in ['CylinderLight', 'DistantLight', 'DomeLight', 'DiskLight', 'GeometryLight', 'RectLight', 'SphereLight']:
            light_num += 1
    return light_num





















# ===========================================
# Functions to process usd prims in the stage
# ===========================================

def IsEmpty(prim):
    # assert prim must be a xform
    assert prim.IsA(UsdGeom.Xform)
    # check if the xform has any children
    children = prim.GetChildren()
    if len(children) == 0:
        return True
    else:
        return False

def IsObjXform(prim):
    if prim.IsA(UsdGeom.Mesh):
        return True
    # check if the xform has any children
    children = prim.GetChildren()
    for child in children:
        if IsObjXform(child):
            return True
    return False

def remove_empty_prims(stage):
    '''
    Remove all empty xforms from the stage.

    Args:
        stage (Usd.Stage): The stage to remove empty xforms from.
    '''
    all_prims = [prim for prim in stage.Traverse() if prim.IsA(UsdGeom.Xform) and prim.GetPath()!= "/World"]
    all_prims.sort(key=lambda x: len(x.GetPath().pathString.split("/")), reverse=True)
    for prim in all_prims:
        if IsEmpty(prim):
            # print(f"Removing empty Xform: {prim.GetPath()}")
            stage.RemovePrim(prim.GetPath())
        else:
            continue

def remove_bad_prims(stage):
    all_prims = [prim for prim in stage.Traverse() if prim.GetPath()!= "/World" and (prim.IsA(UsdGeom.Xform) or prim.IsA(UsdGeom.Mesh))]
    all_prims.sort(key=lambda x: len(x.GetPath().pathString.split("/")), reverse=True)
    # print(f"all_prims: {all_prims}")
    for prim in all_prims:
        # print(f"prim type: {prim.GetTypeName()}")
        if prim.IsA(UsdGeom.Mesh):
            bbox = compute_bbox(prim)
            scale = np.array(bbox.max - bbox.min)
            zero_num = np.sum(np.isclose(scale, 0, atol=1e-2))
            nan_num = np.sum(np.isnan(scale))
            if zero_num > 0 or nan_num > 0:
                print(f"Removing bad prim: {prim.GetPath()}")
                stage.RemovePrim(prim.GetPath())
                continue
        else:
           continue


# ===========================================
# Functions to fix materials
# ===========================================



def read_file(fn):
    with open(fn, 'r') as f:
        return f.read()
    return ''

def write_file(fn, content):
    with open(fn, 'w') as f:
        return f.write(content)

def fix_mdls(usd_path, default_mdl_path):
    base_path, base_name = os.path.split(usd_path)
    stage = Usd.Stage.Open(usd_path)
    pbr_mdl = read_file(default_mdl_path)
    need_to_save = False
    for prim in stage.TraverseAll():
        prim_attrs = prim.GetAttributes()
        for attr in prim_attrs:
            attr_type = attr.GetTypeName()
            if attr_type == "asset":
                attr_name = attr.GetName()
                attr_value = attr.Get()
                
                str_value = str(attr_value)
                if '@' in str_value and len(str_value) > 3:
                    names = str_value.split("@")
                    if names[1] != "OmniPBR.mdl":
                        if "Materials" not in names[1].split("/"):
                            # set new attribute value
                            new_value = "./Materials/" + names[1]
                            if os.path.exists(os.path.join(base_path, new_value)):
                                print(f"Set new value {new_value} to the {attr_name}")
                                attr.Set(new_value)
                                need_to_save = True
                            else:
                                print(f"Cannot find {new_value} in {os.path.join(base_path, './Materials/')}")
                            names[1] = "./Materials/" + names[1]

                        asset_fn = os.path.abspath(os.path.join(base_path, names[1]))
                        # print(asset_fn)
                        if not os.path.exists(asset_fn):
                            print("Find missing file " + asset_fn)
                            fdir, fname = os.path.split(asset_fn)
                            mdl_names = fname.split('.')
                            new_content = pbr_mdl.replace('Material__43', mdl_names[0])
                            write_file(asset_fn, new_content)
                        elif os.path.getsize(asset_fn) < 1:
                            print("Find wrong size file " + asset_fn + ' ' + str(os.path.getsize(asset_fn)))
    if need_to_save:
        stage.Save()



# ======================================
# Functions to convert usd to pointcloud
# ======================================

def to_list(data):
    res = []
    if data is not None:
        res = [_ for _ in data]
    return res

def downsample_point_cloud(point_cloud, target_points):

    if target_points >= point_cloud.shape[0]:
        return point_cloud
    indices = np.random.choice(point_cloud.shape[0], target_points, replace=False)
    downsampled_point_cloud = point_cloud[indices]
    return downsampled_point_cloud

def norm_coords(coords):

    min_coords = np.min(coords, axis=0)
    max_coords = np.max(coords, axis=0)
    norm_coords = (coords - min_coords) / np.max((max_coords - min_coords))
    return norm_coords



def recursive_parse_new(prim):

    points_total = []
    faceVertexCounts_total = []
    faceVertexIndices_total = []

    if prim.IsA(UsdGeom.Mesh):
        prim_imageable = UsdGeom.Imageable(prim)
        xform_world_transform = np.array(
            prim_imageable.ComputeLocalToWorldTransform(Usd.TimeCode.Default())
        )

        points = prim.GetAttribute("points").Get()
        faceVertexCounts = prim.GetAttribute("faceVertexCounts").Get()
        faceVertexIndices = prim.GetAttribute("faceVertexIndices").Get()
        faceVertexCounts = to_list(faceVertexCounts)
        faceVertexIndices = to_list(faceVertexIndices)
        points = to_list(points)
        points = np.array(points)  # Nx3
        ones = np.ones((points.shape[0], 1))  # Nx1
        points_h = np.hstack([points, ones])  # Nx4
        points_transformed_h = np.dot(points_h, xform_world_transform)  # Nx4
        points_transformed = points_transformed_h[:, :3] / points_transformed_h[:, 3][:, np.newaxis]  # Nx3
        points = points_transformed.tolist()
        points = np.array(points)

        if np.isnan(points).any():
            # There is "nan" in points
            print("[INFO] Found NaN in points, performing clean-up...")
            nan_mask = np.isnan(points).any(axis=1)  
            valid_points_mask = ~nan_mask  
            points_clean = points[valid_points_mask].tolist()
            faceVertexIndices = np.array(faceVertexIndices).reshape(-1, 3)
            old_to_new_indices = np.full(points.shape[0], -1)
            old_to_new_indices[valid_points_mask] = np.arange(np.sum(valid_points_mask))
            valid_faces_mask = np.all(old_to_new_indices[faceVertexIndices] != -1, axis=1)
            faceVertexIndices_clean = old_to_new_indices[faceVertexIndices[valid_faces_mask]].flatten().tolist()
            faceVertexCounts_clean = np.array(faceVertexCounts)[valid_faces_mask].tolist()
            base_num = len(points_total)
            faceVertexIndices_total.extend((base_num + np.array(faceVertexIndices_clean)).tolist())
            faceVertexCounts_total.extend(faceVertexCounts_clean)
            points_total.extend(points_clean)
        else:
            base_num = len(points_total)
            faceVertexIndices = np.array(faceVertexIndices)
            faceVertexIndices_total.extend((base_num + faceVertexIndices).tolist())
            faceVertexCounts_total.extend(faceVertexCounts)
            points_total.extend(points)

    children = prim.GetChildren()
    for child in children:
        child_points, child_faceVertexCounts, child_faceVertexIndices = recursive_parse_new(child)
        base_num = len(points_total)
        child_faceVertexIndices = np.array(child_faceVertexIndices)
        faceVertexIndices_total.extend((base_num + child_faceVertexIndices).tolist())
        faceVertexCounts_total.extend(child_faceVertexCounts)
        points_total.extend(child_points)

    return (
        points_total,
        faceVertexCounts_total,
        faceVertexIndices_total,
    )

def get_mesh_from_points_and_faces(points, faceVertexCounts, faceVertexIndices):
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(points)
    triangles = []
    idx = 0
    for count in faceVertexCounts:
        if count == 3:
            triangles.append(faceVertexIndices[idx : idx + 3])
        idx += count
    mesh.triangles = o3d.utility.Vector3iVector(triangles)
    mesh.compute_vertex_normals()
    return mesh

def sample_points_from_mesh(mesh, num_points=1000):
    pcd = mesh.sample_points_uniformly(number_of_points=num_points)
    return pcd

def sample_points_from_prim(prim, num_points=1000):
    points, faceVertexCounts, faceVertexIndices = recursive_parse_new(prim)
    mesh = get_mesh_from_points_and_faces(points, faceVertexCounts, faceVertexIndices)
    pcd = sample_points_from_mesh(mesh, num_points)
    return np.asarray(pcd.points), mesh


# ===============================
# Functions to process pointcloud
# ===============================

def filter_free_noise_v1(pcd):

    def calculate_adaptive_threshold(density_smooth, method='percentile', percentile=5, n_sigma=1):

        non_zero_density = density_smooth[density_smooth > 0]  
        if method == 'percentile':
            threshold = np.percentile(non_zero_density, percentile)  
        elif method == 'sigma':
            mean = np.mean(non_zero_density)
            std = np.std(non_zero_density)
            print("2d mean", mean)
            print("2d std", std)
            # threshold = mean - n_sigma * std  
            threshold = mean if mean < std else mean - n_sigma * std  
        else:
            raise ValueError("Method must be 'percentile' or 'sigma'")
        
        return max(threshold, 0)
    
    points = np.array(pcd.points)
    points_xy = points[:,0:2]
    x_min, x_max = np.min(points_xy[:, 0]), np.max(points_xy[:, 0])
    y_min, y_max = np.min(points_xy[:, 1]), np.max(points_xy[:, 1])
    grid_resolution = 1
    x_bins = np.arange(x_min, x_max + grid_resolution, grid_resolution)
    y_bins = np.arange(y_min, y_max + grid_resolution, grid_resolution)
    density, x_edges, y_edges = np.histogram2d(
        points_xy[:, 0], points_xy[:, 1], bins=(x_bins, y_bins)
    )
    density_smooth = gaussian_filter(density, sigma=1.0)
    
    x_indices = np.digitize(points_xy[:, 0], x_edges) - 1
    y_indices = np.digitize(points_xy[:, 1], y_edges) - 1
    
    density_threshold = calculate_adaptive_threshold(density_smooth, method='sigma')
    print("2d density_threshold", density_threshold)
    mask = np.zeros_like(points_xy[:, 0], dtype=bool)
    for i in range(len(points_xy)):
        if density_smooth[x_indices[i], y_indices[i]] > density_threshold:
            mask[i] = True
    
    filtered_points_xy = points[mask, 0:2]
    filtered_max_x, filtered_min_x = np.max(filtered_points_xy[:, 0]), np.min(filtered_points_xy[:, 0])
    filtered_max_y, filtered_min_y = np.max(filtered_points_xy[:, 1]), np.min(filtered_points_xy[:, 1])

    filtered_scene_points_indices = np.nonzero(
        (points[:, 0] >= filtered_min_x) & (points[:, 0] <= filtered_max_x) & 
        (points[:, 1] >= filtered_min_y) & (points[:, 1] <= filtered_max_y)
    )[0]
    # print(filtered_scene_points_indices)
    pcd.points = o3d.utility.Vector3dVector(points[filtered_scene_points_indices])
    return pcd

def filter_free_noise(pcd, plane='xy', visualize=False):
    points = np.array(pcd.points)
    
    if plane == 'xy':
        points_plane = points[:, 0:2]
    elif plane == 'xz':
        points_plane = np.concatenate([points[:, 0:1], points[:, 2:3]], axis=1)
    else:
        raise ValueError("Invalid plane parameter. Allowed values are 'xy' and 'xz'.")

    min_vals = np.min(points_plane, axis=0)
    max_vals = np.max(points_plane, axis=0)

    grid_resolution = 0.1
    epsilon = 1e-10
    bins = [np.arange(min_val, max_val + grid_resolution + epsilon, grid_resolution) for min_val, max_val in zip(min_vals, max_vals)]

    density, edge1, edge2 = np.histogram2d(
        points_plane[:, 0], points_plane[:, 1], bins=bins
    )
    edges = [edge1, edge2]
    non_zero_density = density > 0
    labeled_array, num_features = label(non_zero_density)

    clusters = [[] for _ in range(num_features)]
    for point in points:
        if plane == 'xy':
            coords = [point[0], point[1]]
        elif plane == 'xz':
            coords = [point[0], point[2]]
        indices = [np.searchsorted(edge, coord, side='right') - 1 for coord, edge in zip(coords, edges)]
        if all(0 <= idx < len(edge) - 1 for idx, edge in zip(indices, edges)):
            label_value = labeled_array[tuple(indices)]
            if label_value > 0:
                clusters[label_value - 1].append(point)

    clusters = [np.array(cluster) for cluster in clusters]
    cluster_densities = []
    cluster_areas = []
    cluster_sizes = []
    for i in range(num_features):
        area = np.sum(labeled_array == (i + 1)) * grid_resolution**2
        cluster_density = len(clusters[i]) / (area + 1e-6)
        cluster_sizes.append(len(clusters[i]))
        cluster_areas.append(area)
        cluster_densities.append(cluster_density)

    normalized_sizes = np.array(cluster_sizes) / np.max(cluster_sizes)
    normalized_areas = np.array(cluster_areas) / np.max(cluster_areas)
    normalized_densities = np.array(cluster_densities) / np.max(cluster_densities)
    weights = {"size": 0.5, "area": 0.3, "density": 0.2}

    combined_scores = (
        weights["size"] * normalized_sizes +
        weights["area"] * normalized_areas +
        weights["density"] * normalized_densities
    )
    print(combined_scores)
    main_cluster_idx = np.argmax(combined_scores)
    pcd.points = o3d.utility.Vector3dVector(clusters[main_cluster_idx])
    if visualize:
        o3d.visualization.draw_geometries([pcd])
    return pcd



def get_z_distribution_coordinate_length(z_points_in_grid, bins):
    hist, bin_edges = np.histogram(z_points_in_grid, bins)
    non_zero_indices = np.where(hist != 0)[0]
    total_length = 0
    for index in non_zero_indices:
        start = bin_edges[index]
        end = bin_edges[index + 1]
        length = end - start
        total_length += length

    return total_length

def filter_extra_ground(pcd, grid_resolution=1):
    points = np.array(pcd.points)
    points_xy = points[:, :2]
    points_z = points[:, 2]

    x_min, x_max = np.min(points_xy[:, 0]), np.max(points_xy[:, 0])
    y_min, y_max = np.min(points_xy[:, 1]), np.max(points_xy[:, 1])
    z_min, z_max = np.min(points_z), np.max(points_z)
    z_resolution = 0.1
    z_max_length = z_max - z_min

    x_bins = np.arange(x_min, x_max + grid_resolution, grid_resolution)
    y_bins = np.arange(y_min, y_max + grid_resolution, grid_resolution)
    z_bins = np.arange(z_min, z_max + z_resolution, z_resolution)
    

    z_ratio = np.zeros((len(x_bins) - 1, len(y_bins) - 1))
    all_z_grid_lengths = []
    all_occupy_ratios = []  


    for i in range(len(x_bins) - 1):
        for j in range(len(y_bins) - 1):
            mask = (points_xy[:, 0] >= x_bins[i]) & (points_xy[:, 0] < x_bins[i + 1]) & \
                   (points_xy[:, 1] >= y_bins[j]) & (points_xy[:, 1] < y_bins[j + 1])
            z_points_in_grid = points_z[mask]
            if len(z_points_in_grid) > 0:
                z_grid_min = np.min(z_points_in_grid)
                z_grid_max = np.max(z_points_in_grid)
                z_grid_length = z_grid_max - z_grid_min
                all_z_grid_lengths.append(z_grid_length)
                z_distribution = get_z_distribution_coordinate_length(z_points_in_grid, z_bins)


                if z_grid_length > 0:
                    occupy_ratio = z_distribution / z_max_length

                else:
                    occupy_ratio = 0.0  
                all_occupy_ratios.append(occupy_ratio)
            else:
                all_z_grid_lengths.append(0)
                all_occupy_ratios.append(0.0)

    max_z_grid_length = np.max(all_z_grid_lengths) if len(all_z_grid_lengths) > 0 else 1

    index = 0
    for i in range(len(x_bins) - 1):
        for j in range(len(y_bins) - 1):
            z_grid_length = all_z_grid_lengths[index]
            occupy_ratio = all_occupy_ratios[index]
            if max_z_grid_length != 0:
                z_ratio[i, j] = z_grid_length / max_z_grid_length
            else:
                z_ratio[i, j] = 0
            index += 1

    keep_mask = np.zeros(len(points), dtype=bool)

    index = 0
    for i in range(len(x_bins) - 1):
        for j in range(len(y_bins) - 1):
            mask = (points_xy[:, 0] >= x_bins[i]) & (points_xy[:, 0] < x_bins[i + 1]) & \
                   (points_xy[:, 1] >= y_bins[j]) & (points_xy[:, 1] < y_bins[j + 1])
            if z_ratio[i, j] >= 0.5 and all_occupy_ratios[index] >= 0.2:
                keep_mask |= mask
            index += 1

    filtered_points = points[keep_mask]
    filtered_points_xy = filtered_points[:, :2]
    filtered_points_max_x, filtered_points_min_x = np.max(filtered_points_xy[:, 0]), np.min(filtered_points_xy[:, 0])
    filtered_points_max_y, filtered_points_min_y = np.max(filtered_points_xy[:, 1]), np.min(filtered_points_xy[:, 1])
    filtered_points = points[(points[:, 0] >= filtered_points_min_x) & (points[:, 0] <= filtered_points_max_x) & 
                            (points[:, 1] >= filtered_points_min_y) & (points[:, 1] <= filtered_points_max_y)]
    # filtered_pcd = o3d.geometry.PointCloud()
    # filtered_pcd.points = o3d.utility.Vector3dVector(filtered_points)
    pcd.points = o3d.utility.Vector3dVector(filtered_points)

    return pcd