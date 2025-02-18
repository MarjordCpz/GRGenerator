import numpy as np
import torch
import cv2
import open3d as o3d
from scipy.spatial.transform import Rotation as R
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks
from sklearn.cluster import DBSCAN
from typing import Union, List
from mathutils import Matrix, Vector, Euler
# Function that calculate pairwise pointcloud distance
def pointcloud_distance(pcdA,pcdB,device='cpu'):
    pointsA = torch.tensor(np.array(pcdA.points),device=device)
    pointsB = torch.tensor(np.array(pcdB.points),device=device)
    cdist = torch.cdist(pointsA,pointsB)
    min_distances1, _ = cdist.min(dim=1)
    return min_distances1.cpu().numpy()

# Function that calculate pairwise pointcloud distance, but ignore the z-dimension
def numpy_2d_distance(AA,BB,device='cpu'):
    pointsA = torch.tensor(np.array(AA),device=device)[:,0:2]
    pointsB = torch.tensor(np.array(BB),device=device)[:,0:2]    
    cdist = torch.cdist(pointsA,pointsB)
    min_distances1, _ = cdist.min(dim=1)
    return min_distances1.cpu().numpy()

# Function that calculate pairwise pointcloud distance, but ignore the z-dimension
def pointcloud_2d_distance(pcdA,pcdB,device='cpu'):
    pointsA = torch.tensor(np.array(pcdA.points),device=device)[:,0:2]
    pointsB = torch.tensor(np.array(pcdB.points),device=device)[:,0:2]
    cdist = torch.cdist(pointsA,pointsB)
    min_distances1, _ = cdist.min(dim=1)
    return min_distances1.cpu().numpy()

# Function that calculate pairwise pointcloud distance, but ignore the z-dimension
def points_2d_distance(pcdA,pcdB,device='cpu'):
    pointsA = torch.tensor(np.array(pcdA),device=device)[:,0:2]
    pointsB = torch.tensor(np.array(pcdB),device=device)[:,0:2]
    cdist = torch.cdist(pointsA,pointsB)
    min_distances1, _ = cdist.min(dim=1)
    return min_distances1.cpu().numpy()

def get_pointcloud_from_depth(rgb:np.ndarray,depth:np.ndarray,intrinsic:np.ndarray,extrinsic:np.ndarray):
    if len(depth.shape) == 3:
        depth = depth[:,:,0]
    filter_z,filter_x = np.where(depth>0)
    depth_values = depth[filter_z,filter_x]
    pixel_z = (depth.shape[0] - 1 - filter_z - intrinsic[1][2]) * depth_values / intrinsic[1][1]
    pixel_x = (filter_x - intrinsic[0][2])*depth_values / intrinsic[0][0]
    pixel_y = depth_values
    color_values = rgb[filter_z,filter_x]
    point_values = np.stack([pixel_x,pixel_z,-pixel_y],axis=-1)
    point_values = np.matmul(extrinsic,np.concatenate((point_values,np.ones((point_values.shape[0],1))),axis=-1).T).T[:,0:3]
    return filter_z,filter_x,depth_values,point_values,color_values

def generate_intrinsic(width,height,hfov,vfov):
    intrinsic = np.eye(3)
    intrinsic[0][0] = width / (2 * (np.tan(np.deg2rad(hfov)/2)))
    intrinsic[1][1] = height / (2 * (np.tan(np.deg2rad(vfov)/2)))
    intrinsic[0][2] = width / 2
    intrinsic[1][2] = height / 2
    return intrinsic

def cpu_pointcloud_from_array(points,colors):
    pointcloud = o3d.geometry.PointCloud()
    pointcloud.points = o3d.utility.Vector3dVector(points)
    pointcloud.colors = o3d.utility.Vector3dVector(colors)
    return pointcloud

def project_to_camera(points,intrinsic,extrinsic):
    inv_extrinsic = np.linalg.inv(extrinsic)
    camera_points = np.concatenate((points,np.ones((points.shape[0],1))),axis=-1)
    camera_points = np.matmul(inv_extrinsic,camera_points.T).T[:,0:3]
    depth_values = -camera_points[:,2]
    filter_x = (camera_points[:,0] * intrinsic[0][0] / depth_values + intrinsic[0][2]).astype(np.int32)
    filter_z = (-camera_points[:,1] * intrinsic[1][1] / depth_values - intrinsic[1][2] + intrinsic[1][2]*2 - 1).astype(np.int32)
    return filter_x,filter_z,depth_values

def project_inview_area(image,points,intrinsic,extrinsic):
    filter_x,filter_z,depth_values = project_to_camera(points,intrinsic,extrinsic)
    area = np.where((filter_x >= 0) & (filter_x < image.shape[1]) & (filter_z >= 0) & (filter_z < image.shape[0]) & (depth_values > 0))[0].shape[0]
    return area

def project_inview_trajectory(image,points,intrinsic,extrinsic):
    filter_x,filter_z,_ = project_to_camera(points * np.array([1,1,0]),intrinsic,extrinsic)
    traj_mask = np.zeros_like(image)
    for i in range(filter_x.shape[0]-1):
        x1,y1,x2,y2 = filter_x[i],filter_z[i],filter_x[i+1],filter_z[i+1]
        if x1 > 0 and x1 < image.shape[1] and y1 > 0 and y1 < image.shape[0] and x2 > 0 and x2 < image.shape[1] and y2 > 0 and y2 < image.shape[0]:
            cv2.line(traj_mask,(filter_x[i],filter_z[i]),(filter_x[i+1],filter_z[i+1]),color=(255,255,255),thickness=10)
    return traj_mask.mean(axis=-1)

def select_view(pcd,intrinsic,extrinsics,width=640,height=480):
    sum_list = []
    for extrinsic in extrinsics:
        xs,zs,ds = project_to_camera(pcd,intrinsic,extrinsic)
        condition = (xs > 0) & (xs < width) & (zs > 0) & (zs < height) & (ds > 0)
        inview_amount = condition.sum()
        sum_list.append(inview_amount)
    return extrinsics[np.argmax(sum_list)]

def clockwise_angle(v1,v2):
    dot_product = np.dot(v1,v2)
    determinat = v1[0]*v2[1] - v1[1]*v2[0]
    angle = np.arctan2(determinat,dot_product)
    if angle < 0:
        angle += 2*np.pi
    return angle

def world2frame(world_R1,world_T1,world_R2,world_T2):
    homo_RT = np.eye(4)
    homo_RT[0:3,0:3] = world_R1
    homo_RT[0:3,3] = world_T1
    R_rel = np.dot(world_R2,world_R1.T)
    T_rel = np.dot(np.linalg.inv(homo_RT),np.array([*world_T2,1]).T)[0:3]
    T_rel[-1] = -T_rel[-1]
    return R_rel,T_rel

def frame2world(base_R1,base_T1,frame_R1,frame_T1):
    homo_RT = np.eye(4)
    homo_RT[0:3,0:3] = base_R1
    homo_RT[0:3,3] = base_T1
    frame_T1[-1] = -frame_T1[-1]
    world_R = np.dot(frame_R1,base_R1)
    world_T = np.dot(homo_RT,np.array([*frame_T1,1])).T[0:3]
    return world_R,world_T 

def clockwise_angle(v1,v2):
    dot_product = np.dot(v1,v2)
    determinat = v1[0]*v2[1] - v1[1]*v2[0]
    angle = np.arctan2(determinat,dot_product)
    if angle < 0:
        angle += 2*np.pi
    return angle


def build_transformation_mat(translation,rotation):
    translation = np.array(translation)
    rotation = np.array(rotation)

    mat = np.eye(4)
    if translation.shape[0] == 3:
        mat[:3, 3] = translation
    else:
        raise RuntimeError(f"Translation has invalid shape: {translation.shape}. Must be (3,) or (3,1) vector.")
    if rotation.shape == (3, 3):
        mat[:3, :3] = rotation
    elif rotation.shape[0] == 3:
        mat[:3, :3] = np.array(R.from_euler('xyz',rotation).as_matrix())
    else:
        raise RuntimeError(f"Rotation has invalid shape: {rotation.shape}. Must be rotation matrix of shape "
                           f"(3,3) or Euler angles of shape (3,) or (3,1).")

    return mat


import math

def getProjectionMatrix(znear, zfar, fovX, fovY):
    tanHalfFovY = math.tan((fovY / 2))
    tanHalfFovX = math.tan((fovX / 2))
    top = tanHalfFovY * znear
    bottom = -top
    right = tanHalfFovX * znear
    left = -right
    P = torch.zeros((4,4),dtype=torch.float,device="cuda")
    z_sign = 1.0
    P[0, 0] = 2.0 * znear / (right - left)
    P[1, 1] = 2.0 * znear / (top - bottom)
    P[0, 2] = (right + left) / (right - left)
    P[1, 2] = (top + bottom) / (top - bottom)
    P[3, 2] = z_sign
    P[2, 2] = z_sign * zfar / (zfar - znear)
    P[2, 3] = -(zfar * znear) / (zfar - znear)
    return P

def focal2fov(focal, pixels):
    return 2*math.atan(pixels/(2*focal))

def extract_floor_heights(scene_pcd_points):

    def filter(z_hist):
        # filter the histogram
        threshold = np.mean(z_hist[0])

        left_bound = 0
        for i in range(len(z_hist[0])):
            if z_hist[0][i] > threshold:
                left_bound = i - 1 if i > 0 else 0
                break

        right_bound = len(z_hist[0]) - 1
        for i in range(len(z_hist[0]) - 1, -1, -1):
            if z_hist[0][i] > threshold:
                right_bound = i + 1 if i < len(z_hist[0]) - 2 else len(z_hist[0]) - 2
                break
            
        z_hist_filtered = z_hist[0][left_bound:right_bound + 1]
        z_hist_bins_filtered = z_hist[1][left_bound:right_bound + 2]

        return (z_hist_filtered, z_hist_bins_filtered)


    z_coords = scene_pcd_points[:,2]
    reselotion = 0.01
    bins = np.abs(np.max(z_coords) - np.min(z_coords)) / reselotion
    print("bins", bins)
    z_hist = np.histogram(z_coords, bins=int(bins))
    z_hist = filter(z_hist)
    # smooth the histogram
    z_hist_smooth = gaussian_filter1d(z_hist[0], sigma=1)
    # Find the peaks in this histogram.
    distance = 0.2 / reselotion
    print("distance", distance)
    # set the min peak height based on the histogram
    # print(np.mean(z_hist_smooth))
    min_peak_height = np.percentile(z_hist_smooth, 90)
    print("min_peak_height", min_peak_height)
    peaks, _ = find_peaks(z_hist_smooth, distance=distance, height=min_peak_height)
    import matplotlib.pyplot as plt
    plt.figure()
    plt.plot(z_hist[1][:-1], z_hist_smooth)
    plt.plot(z_hist[1][peaks], z_hist_smooth[peaks], "x")
    plt.hlines(
        min_peak_height, np.min(z_hist[1]), np.max(z_hist[1]), colors="r"
    )
    plt.savefig("floor_histogram.png")
    peaks_locations = z_hist[1][peaks]
    clustering = DBSCAN(eps=1, min_samples=1).fit(peaks_locations.reshape(-1, 1))
    labels = clustering.labels_

    plt.figure()
    plt.plot(z_hist[1][:-1], z_hist_smooth)
    plt.plot(z_hist[1][peaks], z_hist_smooth[peaks], "x")
    plt.hlines(
        min_peak_height, np.min(z_hist[1]), np.max(z_hist[1]), colors="r"
    )
    # plot the clusters
    for i in range(len(np.unique(labels))):
        plt.plot(
            z_hist[1][peaks[labels == i]],
            z_hist_smooth[peaks[labels == i]],
            "o",
        )
    plt.savefig("floor_histogram_cluster.png")

    # for each cluster find the top 2 peaks
    clustred_peaks = []
    for i in range(len(np.unique(labels))):
        # for first and last cluster, find the top 1 peak
        if i == 0 or i == len(np.unique(labels)) - 1:
            p = peaks[labels == i]
            top_p = p[np.argsort(z_hist_smooth[p])[-1:]].tolist()
            top_p = [z_hist[1][p] for p in top_p]
            clustred_peaks.append(top_p)
            continue
        p = peaks[labels == i]
        top_p = p[np.argsort(z_hist_smooth[p])[-2:]].tolist()
        top_p = [z_hist[1][p] for p in top_p]
        clustred_peaks.append(top_p)
    clustred_peaks = [item for sublist in clustred_peaks for item in sublist]
    clustred_peaks = np.sort(clustred_peaks)
    print("clustred_peaks", clustred_peaks)

    floors = []
    # for every two consecutive peaks with 2m distance, assign floor level
    for i in range(0, len(clustred_peaks) - 1, 2):
        floors.append([clustred_peaks[i], clustred_peaks[i + 1]])
    print("floors", floors)
    return floors

def build_transformation_mat(translation: Union[np.ndarray, List[float], Vector],
                             rotation: Union[np.ndarray, List[List[float]], Matrix]) -> np.ndarray:
    """ Build a transformation matrix from translation and rotation parts.

    :param translation: A (3,) vector representing the translation part.
    :param rotation: A 3x3 rotation matrix or Euler angles of shape (3,).
    :return: The 4x4 transformation matrix.
    """
    translation = np.array(translation)
    rotation = np.array(rotation)

    mat = np.eye(4)
    if translation.shape[0] == 3:
        mat[:3, 3] = translation
    else:
        raise RuntimeError(f"Translation has invalid shape: {translation.shape}. Must be (3,) or (3,1) vector.")
    if rotation.shape == (3, 3):
        mat[:3, :3] = rotation
    elif rotation.shape[0] == 3:
        mat[:3, :3] = np.array(Euler(rotation).to_matrix())
    else:
        raise RuntimeError(f"Rotation has invalid shape: {rotation.shape}. Must be rotation matrix of shape "
                           f"(3,3) or Euler angles of shape (3,) or (3,1).")

    return mat