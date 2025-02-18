import numpy as np
import cv2
import matplotlib.pyplot as plt
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from geometry_tools import *
from scipy.interpolate import CubicSpline
from pathfinding.core.diagonal_movement import DiagonalMovement
from pathfinding.core.grid import Grid
from pathfinding.finder.a_star import AStarFinder

# This is a vanilla path planner used to generate collision-free trajectory when navigation in an indoor scene #
# The input should be a point-cloud, and it filter the navigable areas and obstacle areas with respect to the z-axis coordinates #
# The path planning is divided into three steps:
# (1) Generate the ESDF based on the point-cloud, TODO: Change into Voxel Map
# (2) Discrete the point-cloud into a grid-map with respect to the <grid_resolution>, and perform A-star to plan a rough path
# (3) Fine-Grain the A-star waypoint with a greedy-search in local areas
# (4) Use interpolation to smooth the trajectories and down-sample points considering the <sim_dt> and <robot_lin speed>, <robot_ang_speed>
class PathPlanner:
    def __init__(self,ceiling_offset=1.8,grid_resolution=0.25,safe_distance=0.3,robot_lin_speed=0.8,robot_ang_speed=1.0,sim_dt=0.05):
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
        navigable_voxelcost[(navigable_voxelcost > 0) & (navigable_voxelcost <= 0.5)] = 10
        navigable_voxelcost[(navigable_voxelcost > 0.5) & (navigable_voxelcost <= 1.0)] = 1
        color_navigable_costmap = cv2.applyColorMap(((navigable_voxelcost/navigable_voxels.max())*255).astype(np.uint8),cv2.COLORMAP_JET)
        color_navigable_costmap = cv2.resize(color_navigable_costmap,(0,0),fx=10,fy=10,interpolation=cv2.INTER_NEAREST)
        return navigable_voxelcost.astype(np.float32),color_navigable_costmap
    
    def reset(self,floor_height,robot_height,scene_pcd):
        scene_points = np.array(scene_pcd.points)
        self.robot_height = robot_height
        self.scene_pcd = scene_pcd.select_by_index(np.nonzero((scene_points[:,2] < floor_height + self.ceiling_offset) & (scene_points[:,2] > floor_height - 0.1))[0])
        scene_points = np.array(self.scene_pcd.points)
        self.navigable_pcd = self.scene_pcd.select_by_index(np.nonzero((scene_points[:,2] < floor_height + 0.1) & (scene_points[:,2] > floor_height - 0.1))[0])
        self.obstacle_pcd = self.scene_pcd.select_by_index(np.nonzero((scene_points[:,2] > floor_height + 0.1) & (scene_points[:,2] < 3*robot_height))[0])
        self.obstacle_pcd.colors = o3d.utility.Vector3dVector(np.array(self.obstacle_pcd.points)*0.0 + 0.6)
        self.obstacle_distance = pointcloud_2d_distance(self.navigable_pcd,self.obstacle_pcd.voxel_down_sample(self.grid_resolution))
        # smaller distance -> smaller value
        self.safe_value = (1 - (self.obstacle_distance < self.safe_distance)) * (self.obstacle_distance / (self.obstacle_distance.max() + 1e-6))
        self.safe_value = np.clip(self.safe_value,0,0.5)
        self.cost_color = plt.get_cmap("jet")(self.safe_value)[:,0:3]
        self.navigable_pcd.colors = o3d.utility.Vector3dVector(self.cost_color)
        self.max_bound = scene_points.max(axis=0)
        self.min_bound = scene_points.min(axis=0)
        self.decision_value = np.array(self.safe_value)
        self.decision_map,self.color_decision_map = self.point_to_map(np.array(self.navigable_pcd.points),self.decision_value)
    
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
        navigable_points = np.array(self.navigable_pcd.points)
        # sample-based optimize the result point to get better results
        for i in range(result_point.shape[0]):
            try:
                local_distance = points_2d_distance(navigable_points,np.array([[result_point[i,0],result_point[i,1],0.0]]))
                local_indexes = np.where(local_distance < self.grid_resolution)[0]
                local_optima = navigable_points[local_indexes[np.argmax(self.safe_value[local_indexes])]]
                result_point[i][0] = local_optima[0]
                result_point[i][1] = local_optima[1]
            except:
                pass
        
        downsample_index = [0]
        for i in range(result_point.shape[0]):
            distance = np.abs(result_point[i][0] - result_point[downsample_index[-1]][0]) + np.abs(result_point[i][1] - result_point[downsample_index[-1]][1])
            if distance > 1.5:
                downsample_index.append(i)
        result_point = result_point[downsample_index]
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

    def generate_trajectory(self,start_point,end_point):
        status,waypoints = self.astar_waypoint(start_point,end_point)
        if status == False:
            return False,[],[]
        t = np.linspace(0,1,waypoints.shape[0])
        cs_x = CubicSpline(t,waypoints[:,0])
        cs_y = CubicSpline(t,waypoints[:,1])
        t_fine = np.linspace(0,1,waypoints.shape[0]*100)
        x_fine = cs_x(t_fine)
        y_fine = cs_y(t_fine)
        result_points = np.stack((x_fine,y_fine),axis=-1)
        result_rotations = self.astar_rotation(result_points)
        # downsample points
        downsample_index = [0]
        for i in range(result_points.shape[0]):
            pose_distance = np.abs(result_points[i][0] - result_points[downsample_index[-1]][0]) + np.abs(result_points[i][1] - result_points[downsample_index[-1]][1])
            rot_distance = min(abs(result_rotations[i]-result_rotations[downsample_index[-1]]),2*np.pi - abs(result_rotations[i]-result_rotations[downsample_index[-1]]))
            if pose_distance > (self.sim_dt * self.robot_lin_speed) or rot_distance > (self.sim_dt * self.robot_ang_speed):
                downsample_index.append(i)
        result_points = np.array(result_points)[downsample_index]
        result_rotations = np.array(result_rotations)[downsample_index]
        return True,result_points,result_rotations
    
    



