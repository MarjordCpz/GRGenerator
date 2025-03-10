import numpy as np
import cv2
import matplotlib.pyplot as plt
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from geometry_tools import *
from scipy.interpolate import CubicSpline
from skimage.morphology import dilation, erosion
from pathfinding.core.diagonal_movement import DiagonalMovement
from pathfinding.core.grid import Grid
from pathfinding.finder.a_star import AStarFinder
from path_optimizer import optimize_trajectory
from scipy.interpolate import splprep, splev
from scipy.spatial import ConvexHull
from shapely.geometry import Polygon


class PathPlanner:
    def __init__(self,ceiling_offset=1.8,grid_resolution=0.05,safe_distance=0.3,robot_lin_speed=0.8,robot_ang_speed=1.0,sim_dt=0.05):
        self.ceiling_offset = ceiling_offset
        self.grid_resolution = grid_resolution
        self.safe_distance = safe_distance
        self.robot_lin_speed = robot_lin_speed
        self.robot_ang_speed = robot_ang_speed
        self.sim_dt = sim_dt
        self.finder = AStarFinder(diagonal_movement=DiagonalMovement.always)
    
    def point_to_map(self,points,values):
        grid_dimension = np.ceil((self.max_bound-self.min_bound) / self.grid_resolution).astype(int)
        grid_dimension[2] = max(grid_dimension[2],1)
        navigable_voxels = 10*np.ones(grid_dimension,dtype=np.float32)
        navigable_indices = np.floor((points - self.min_bound) / self.grid_resolution).astype(int)
        navigable_voxels[navigable_indices[:,0],navigable_indices[:,1],navigable_indices[:,2]] = values
        navigable_voxelcost = navigable_voxels.min(axis=2)
        navigable_voxelcost[(navigable_voxelcost == 0) | (navigable_voxelcost > 5)] = 0 # not navigable points
        navigable_voxelcost[(navigable_voxelcost > 0)] = 1

        def filter_array(arr):
            dilated = dilation(arr)
            dilated = dilation(dilated)
            eroded = erosion(dilated)
            return eroded
        navigable_voxelcost = filter_array(navigable_voxelcost)
        color_navigable_costmap = cv2.applyColorMap(((navigable_voxelcost/navigable_voxels.max())*255).astype(np.uint8),cv2.COLORMAP_JET)
        color_navigable_costmap = cv2.resize(color_navigable_costmap,(0,0),fx=10,fy=10,interpolation=cv2.INTER_NEAREST_EXACT)
        return navigable_voxelcost.astype(np.float32),color_navigable_costmap
    
    def map_to_points(self,obstacle_map):
        # 把栅格化后的所有点转回点云，这里已经实现了栅格图的滤波
        obstacle_idx = np.where(obstacle_map == 0)
        navigable_idx = np.where(obstacle_map > 0)
        obstacle_grid_y, obstacle_grid_x = obstacle_idx
        obstacle_grid_coords = np.column_stack((obstacle_grid_x, obstacle_grid_y)).astype(np.float32)

        navigable_grid_y, navigable_grid_x = navigable_idx
        navigable_grid_coords = np.column_stack((navigable_grid_x, navigable_grid_y)).astype(np.float32)
 

        obstacle_pts = obstacle_grid_coords * self.grid_resolution + self.min_bound[:2]
        navigable_pts = navigable_grid_coords * self.grid_resolution + self.min_bound[:2]

        return obstacle_pts, navigable_pts, obstacle_idx, navigable_idx
    
    def to_esdf(self,obstacle_map):
        obstacle_pts, navigable_pts, obstacle_idx, navigable_idx = self.map_to_points(obstacle_map)
        # navi_idx_x = navigable_pts[:,]
        esdf_height, esdf_width = obstacle_map.shape
        esdf = np.zeros((esdf_height, esdf_width))
        # 1. 先计算障碍物距离
        safe_distance = numpy_2d_distance(navigable_pts, obstacle_pts)
        esdf[navigable_idx] = safe_distance
        # 2. 再计算障礙物到free的距離
        obs_to_free_distance = numpy_2d_distance(obstacle_pts, navigable_pts)
        # esdf[obstacle_idx] = -obs_to_free_distance
        max_distance_visual = 1.2
        
        self.esdf_color = cv2.applyColorMap(((np.clip(esdf, 0, max_distance_visual) / max_distance_visual) * 255).astype(np.uint8), cv2.COLORMAP_JET)
        self.esdf_color = cv2.resize(self.esdf_color,(0,0),fx=10,fy=10,interpolation=cv2.INTER_LINEAR)
        esdf[obstacle_idx] = -obs_to_free_distance
        self.esdf = esdf

    def reset(self,floor_height,robot_height,scene_pcd):
        scene_points = np.array(scene_pcd.points)
        self.robot_height = robot_height
        self.scene_pcd = scene_pcd.select_by_index(np.nonzero((scene_points[:,2] < floor_height + self.ceiling_offset) & (scene_points[:,2] > floor_height - 0.1))[0])
        scene_points = np.array(self.scene_pcd.points)
        self.navigable_pcd = self.scene_pcd.select_by_index(np.nonzero((scene_points[:,2] < floor_height + 0.1) & (scene_points[:,2] > floor_height - 0.1))[0])
        self.obstacle_pcd = self.scene_pcd.select_by_index(np.nonzero((scene_points[:,2] > floor_height + 0.2) & (scene_points[:,2] < 2*robot_height))[0])
        self.obstacle_distance = pointcloud_2d_distance(self.navigable_pcd,self.obstacle_pcd.voxel_down_sample(self.grid_resolution))
        # smaller distance -> smaller value
        self.safe_value = (1 - (self.obstacle_distance < self.safe_distance)) * (self.obstacle_distance / (self.obstacle_distance.max() + 1e-6))
        self.cost_color = plt.get_cmap("jet")(self.safe_value)[:,0:3]
        self.navigable_pcd.colors = o3d.utility.Vector3dVector(self.cost_color)
        self.max_bound = scene_points.max(axis=0)
        self.min_bound = scene_points.min(axis=0)
        self.decision_value = np.array(self.safe_value)
        self.decision_map,self.color_decision_map = self.point_to_map(np.array(self.navigable_pcd.points),self.decision_value)
        self.to_esdf(self.decision_map)
        self.navigable_clusters = self.cluster_navigable_points(visualize=False)
        self.clusters_score = []
        for cluster in self.navigable_clusters:
            cluster_points = np.array(cluster.points)
            self.clusters_score.append(cluster_points.shape[0])
        self.clusters_score = np.array(self.clusters_score)
        self.clusters_score = (self.clusters_score / self.clusters_score.max() + 1) 

    def cluster_navigable_points(self, visualize=True):
        available_points = self.navigable_pcd.select_by_index(np.where(self.safe_value > 0.1)[0])
        print("available points:",np.array(available_points.points).shape[0])
        wall_points = self.navigable_pcd.select_by_index(np.where(self.safe_value < 0.1)[0])
        wall_2d_hull = ConvexHull(np.array(wall_points.points)[:,0:2])
        wall_2d_polygon = Polygon(np.array(wall_points.points)[:,0:2][wall_2d_hull.vertices])
        labels = np.array(available_points.cluster_dbscan(eps=4*self.grid_resolution,min_points=20))

        if visualize:
            unique_labels = set(labels)
            colors = [plt.cm.Spectral(each) for each in np.linspace(0, 1, len(unique_labels))]
            plt.figure(figsize=(10, 8))
            for k, col in zip(unique_labels, colors):
                if k == -1:
                    col = [0, 0, 0, 1]
                class_member_mask = (labels == k)
                xy = np.array(available_points.points)[:,0:2][class_member_mask]
                plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
                        markeredgecolor='k', markersize=6)

            plt.title('2D Point Cloud Clustering (DBSCAN)')
            plt.xlabel('X')
            plt.ylabel('Y')
            plt.grid(True)
            plt.savefig('clustering_visualization.png')
        debug_pointcloud = o3d.geometry.PointCloud()
        cluster_navigable_points = []
        for i,label in enumerate(np.unique(labels)):
            color = np.random.rand(1,3)
            cluster_points = available_points.select_by_index(np.where(labels == label)[0])
            print(label)
            print(np.array(cluster_points.points).shape[0])
            if np.array(cluster_points.points).shape[0] <= 3:
                continue
            navi_2d_hull = ConvexHull(np.array(cluster_points.points)[:,0:2])
            navi_2d_polygon = Polygon(np.array(cluster_points.points)[:,0:2][navi_2d_hull.vertices])
            if wall_2d_polygon.contains(navi_2d_polygon) and np.array(cluster_points.points).shape[0] > 1000: 
                cluster_points.colors = o3d.utility.Vector3dVector(np.tile(color,(len(cluster_points.points),1)))
                debug_pointcloud = debug_pointcloud + cluster_points
                cluster_navigable_points.append(cluster_points)
        return cluster_navigable_points

    def start_end_sampler(self, minimum_distance=5.0):
        cluster_index = np.random.choice(len(self.navigable_clusters),p=np.exp(self.clusters_score/0.2)/np.sum(np.exp(self.clusters_score/0.2)))
        cluster_points = np.array(self.navigable_clusters[cluster_index].points)
        cluster_points_max, cluster_points_min = cluster_points.max(axis=0), cluster_points.min(axis=0)
        cluster_max_distance = np.linalg.norm(cluster_points_max - cluster_points_min)
        print("area distance:", cluster_max_distance)
        minimum_distance = min(minimum_distance, cluster_max_distance - 1)
        start_point = cluster_points[np.random.choice(cluster_points.shape[0])]
        distance = points_2d_distance(cluster_points,np.array([start_point]))
        possible_index = np.where(distance > minimum_distance)[0]
        if len(possible_index) == 0:
            return False,[],[]
        else:
            return True,start_point,cluster_points[np.random.choice(possible_index)]
        

    def point_to_grid(self,point):
        if len(point.shape) == 1:
            grid_index = np.floor((point[0:2] - self.min_bound[0:2])/self.grid_resolution).astype(np.int32)
        else:
            grid_index = np.floor((point[:,0:2] - self.min_bound[0:2])/self.grid_resolution).astype(np.int32)
        return grid_index
    
    def grid_to_point(self,grid):
        if len(grid.shape) == 1:
            grid = np.array([*grid,1],np.float32)
            point = grid * self.grid_resolution + self.min_bound
        else:
            grid = np.concatenate((grid,np.ones((grid.shape[0],1))),axis=-1)
            point = grid * self.grid_resolution + self.min_bound[np.newaxis,:]
        return point
    
    def astar_waypoint(self,start_point,end_point):
        start_index = self.point_to_grid(start_point)
        end_index = self.point_to_grid(end_point)
        if self.decision_map[end_index[0],end_index[1]] == 0:
            navigable_ys,navigable_xs = np.where(self.decision_map>0)
            navigable_dist = np.sqrt(np.square(navigable_ys - end_index[0]) + np.square(navigable_xs - end_index[1]))
            end_index[0] = navigable_ys[navigable_dist.argmin()]
            end_index[1] = navigable_xs[navigable_dist.argmin()]
        grid = Grid(matrix=self.decision_map)
        start = grid.node(start_index[1],start_index[0])
        goal = grid.node(end_index[1],end_index[0])
        path,status = self.finder.find_path(start,goal,grid)
        result_path = [[p.y,p.x] for p in path]
        result_point = self.grid_to_point(np.array(result_path))
        if len(result_path) < 1:
            return False,[]
        self.initial_pts = np.copy(result_point)

        return True,result_point

    def astar_rotation(self,waypoints):
        target_point = waypoints[1:][:,0:2]
        start_point = waypoints[:-1][:,0:2]
        vectors = target_point - start_point
        angles = []
        for v in vectors:
            angle = 2*np.pi - clockwise_angle(v,np.array([0,1]))
            angles.append(angle)
        angles.append(angles[-1])
        return angles

    def interpolate_angles(self, t, key_angles, t_fine):
        interpolated_angles = []
        for i in range(len(t) - 1):
            start_index = np.searchsorted(t_fine, t[i])
            end_index = np.searchsorted(t_fine, t[i + 1])
            num_points = end_index - start_index
            if num_points > 0:
                angle_start = key_angles[i]
                angle_end = key_angles[i + 1]
                diff = angle_end - angle_start
                if diff > np.pi:
                    diff -= 2 * np.pi
                elif diff < -np.pi:
                    diff += 2 * np.pi
                interpolated = angle_start + np.linspace(0, diff, num_points)
                interpolated = (interpolated + np.pi) % (2 * np.pi) - np.pi
                interpolated_angles.extend(interpolated)
                
        interpolated_angles.append(key_angles[-1])
        return interpolated_angles

    def generate_trajectory(self,start_point,end_point, start_angle=None):
        # range of start angle should be [-pi, pi]
        status,waypoints = self.astar_waypoint(start_point,end_point)
        if status == False:
            return False,[],[]
        rotate_in_place = False
        # waypoints = self.smooth_path(waypoints, smooth_factor=0.5, num_points=waypoints.shape[0])
        initial_points = waypoints[::3]

        # decide whether to rotate in place
        if start_angle == None:
            optimized_points = optimize_trajectory(initial_points[:, :2], self.esdf, 0.3, self.min_bound[0], self.max_bound[0], self.min_bound[1], self.max_bound[1], self.grid_resolution, 0.5)
        else:
            # rotation in place when the angle is in FOV. Then plan the path
            FOV = np.pi / 8.0
            astar_angle = np.arctan2(initial_points[1, 1] - initial_points[0, 1], initial_points[1, 0] - initial_points[0, 0])
            if -FOV < angle_difference(astar_angle, start_angle) < FOV:
                optimized_points = optimize_trajectory(initial_points[:, :2], self.esdf, 0.4, self.min_bound[0], self.max_bound[0], self.min_bound[1], self.max_bound[1], self.grid_resolution, 0.5, start_angle)
            else:
                rotate_in_place = True
                # judge the rotation direction
                candidate_angle1 = angle_add(astar_angle, FOV)
                candidate_angle2 = angle_add(astar_angle, -FOV)
                if abs(angle_difference(start_angle, candidate_angle1)) < abs(angle_difference(start_angle, candidate_angle2)):
                    rotated_desired_angle = candidate_angle1
                else:
                    rotated_desired_angle = candidate_angle2
                optimized_points = optimize_trajectory(initial_points[:, :2], self.esdf, 0.3, self.min_bound[0], self.max_bound[0], self.min_bound[1], self.max_bound[1], self.grid_resolution, 0.5, rotated_desired_angle)

        t = np.linspace(0,1,optimized_points.shape[0])
        cs_x = CubicSpline(t,optimized_points[:,0])
        cs_y = CubicSpline(t,optimized_points[:,1])
        t_fine = np.linspace(0,1,optimized_points.shape[0]*100)
        self.x_fine = cs_x(t_fine)
        self.y_fine = cs_y(t_fine)
        optimized_rotations = self.astar_rotation(optimized_points)
        result_rotations = self.interpolate_angles(t, optimized_rotations, t_fine)
        result_points = np.stack((self.x_fine,self.y_fine),axis=-1)
        # result_rotations = self.astar_rotation(result_points)
        # only for optimization comparison
        t_init = np.linspace(0,1,self.initial_pts.shape[0])
        cs_x_init = CubicSpline(t_init,self.initial_pts[:,0])
        cs_y_init = CubicSpline(t_init,self.initial_pts[:,1])
        t_fine_init = np.linspace(0,1,self.initial_pts.shape[0]*100)
        self.x_fine_init = cs_x_init(t_fine_init)
        self.y_fine_init = cs_y_init(t_fine_init)

        # fake a controller to generate the camera pose for trainning
        final_points = []
        final_rotations = []
        if rotate_in_place:
            target_angle = start_angle - np.pi / 4
            clockwise_diff = (target_angle - result_rotations[0] + np.pi * 2) % (2 * np.pi)
            counterclockwise_diff = (result_rotations[0] - target_angle + np.pi * 2) % (2 * np.pi)          
            if clockwise_diff < counterclockwise_diff:
                cur_angle = start_angle - np.pi / 4
                while clockwise_diff > self.robot_ang_speed * self.sim_dt:
                    cur_angle = angle_add(cur_angle, - self.robot_ang_speed * self.sim_dt)
                    clockwise_diff -= self.robot_ang_speed * self.sim_dt
                    final_points.append(result_points[0])
                    final_rotations.append(normalized_angle(cur_angle))
            else:
                cur_angle = start_angle - np.pi / 4
                while counterclockwise_diff > self.robot_ang_speed * self.sim_dt:
                    cur_angle = angle_add(cur_angle, self.robot_ang_speed * self.sim_dt)
                    counterclockwise_diff -= self.robot_ang_speed * self.sim_dt
                    final_points.append(result_points[0])
                    final_rotations.append(normalized_angle(cur_angle))     

        final_points.append(result_points[0])
        final_rotations.append(result_rotations[0])        
        for i in range(result_points.shape[0]):
            pose_distance = np.abs(result_points[i][0] - final_points[-1][0]) + np.abs(result_points[i][1] - final_points[-1][1])
            rot_distance = min(abs(result_rotations[i]-final_rotations[-1]),2*np.pi - abs(result_rotations[i]-final_rotations[-1]))
            if pose_distance > (self.sim_dt * self.robot_lin_speed) or rot_distance > (self.sim_dt * self.robot_ang_speed):
                final_points.append(result_points[i])
                final_rotations.append(result_rotations[i])

        self.start_angle = start_angle
        return True,np.array(final_points),np.array(final_rotations)
    
    def visualize_trajectory_on_map(self, result_points):
        visualized_map = cv2.applyColorMap(((self.decision_map / self.decision_map.max()) * 255).astype(np.uint8),
                                                cv2.COLORMAP_JET)
        visualized_map = cv2.resize(visualized_map,(0,0),fx=10,fy=10,interpolation=cv2.INTER_NEAREST_EXACT)

        for i in range(len(self.x_fine)):
            grid_index_x = (np.floor((self.x_fine[i] - self.min_bound[0]) * 10 / self.grid_resolution)).astype(np.int32)
            grid_index_y = (np.floor((self.y_fine[i] - self.min_bound[1]) * 10 / self.grid_resolution)).astype(np.int32)
            if 0 <= grid_index_y < visualized_map.shape[1] and 0 <= grid_index_x < visualized_map.shape[0]:
                cv2.circle(visualized_map, (grid_index_y, grid_index_x), 2, [0, 255, 0], 2)
            if i == 0:
                cv2.drawMarker(visualized_map, (grid_index_y, grid_index_x), (0, 255, 255), markerType=cv2.MARKER_STAR, markerSize=16, thickness=2)
                if self.start_angle != None:
                    cv2.arrowedLine(visualized_map, (grid_index_y, grid_index_x), (int(grid_index_y + np.sin(self.start_angle) * 50), int(grid_index_x + np.cos(self.start_angle) * 50)), (255, 255, 0), 10, cv2.LINE_AA, 0, 0.3)
        for i in range(len(self.x_fine_init)):
            grid_index_x = (np.floor((self.x_fine_init[i] - self.min_bound[0]) * 10 / self.grid_resolution)).astype(np.int32)
            grid_index_y = (np.floor((self.y_fine_init[i] - self.min_bound[1]) * 10 / self.grid_resolution)).astype(np.int32)
            if 0 <= grid_index_y < visualized_map.shape[1] and 0 <= grid_index_x < visualized_map.shape[0]:
                cv2.circle(visualized_map, (grid_index_y, grid_index_x), 2, (255, 0, 255), 2)
            if i == 0:
                cv2.drawMarker(visualized_map, (grid_index_y, grid_index_x), (255, 0, 255), markerType=cv2.MARKER_STAR, markerSize=16, thickness=2) 

        return visualized_map
