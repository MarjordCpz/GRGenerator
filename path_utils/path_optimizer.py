import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt

def distance_to_obstacles(traj, esdf_map, x_min, x_max, y_min, y_max, resolution, clearance):
    cost = 0
    for point in traj:
        x_index = int(np.floor((point[0] - x_min) / resolution)) 
        y_index = int(np.floor((point[1] - y_min) / resolution))
        
        diff = (point - np.array([x_min + resolution * x_index, y_min + resolution * y_index]))
        if 0 < x_index < esdf_map.shape[0] - 1 and 0 < y_index < esdf_map.shape[1] - 1:
            # 做一个双线性插值
            values = np.zeros((2, 2))
            for x in range(2):
                for y in range(2):
                    values[x][y] = esdf_map[x_index + x, y_index + y]
            v00 = (1 - diff[0]) * values[0][0] + diff[0] * values[1][0]
            v01 = (1 - diff[0]) * values[0][1] + diff[0] * values[1][1]
            dist = (1 - diff[1]) * v00 + diff[1] * v01
            if dist < clearance:
                cost += (clearance - dist) ** 2
    return cost

def gradient_objective_function(traj_flattened, esdf_map, lambda_smooth, lambda_start, x_min, x_max, y_min, y_max, resolution, clearance, attract_point):
    traj = traj_flattened.reshape(-1, 2)
    n = traj.shape[0]
    grad_distance = np.zeros((n, 2))
    grad_smoothness = np.zeros((n, 2))
    grad_start = np.zeros((n, 2))
    # 避障的梯度
    for i in range(len(traj) - 2):
        point = traj[i + 1]
        # 不优化起点和终点
        x_index = int(np.floor((point[0] - x_min) / resolution)) 
        y_index = int(np.floor((point[1] - y_min) / resolution))
        
        diff = (point - np.array([x_min + resolution * x_index, y_min + resolution * y_index]))
        if 0 < x_index < esdf_map.shape[0] - 1 and 0 < y_index < esdf_map.shape[1] - 1:
            # 做一个双线性插值
            values = np.zeros((2, 2))
            for x in range(2):
                for y in range(2):
                    values[x][y] = esdf_map[x_index + x, y_index + y]
            v00 = (1 - diff[0]) * values[0][0] + diff[0] * values[1][0]
            v01 = (1 - diff[0]) * values[0][1] + diff[0] * values[1][1]
            dist = (1 - diff[1]) * v00 + diff[1] * v01
            if dist < clearance:
                grad_distance[i + 1 ,1] = -2.0 * (clearance - dist) * (v01 - v00)
                grad_distance[i + 1 ,0] = -2.0 * (clearance - dist) * ((1 - diff[1]) * (values[1][0] - values[0][0]) + diff[1] * (values[1][1] - values[0][1]))

    # 计算平滑度代价的梯度
    if n > 1:
        diff_x_1 = np.diff(traj[:, 0], 1)
        diff_y_1 = np.diff(traj[:, 1], 1)
        diff_x_2 = np.diff(traj[:, 0], 2)
        diff_y_2 = np.diff(traj[:, 1], 2)

        for i in range(1, n - 1):
            grad_smoothness[i, 0] = -4 * diff_x_2[i - 1]
            grad_smoothness[i, 1] = -4 * diff_y_2[i - 1]
    else:
        grad_smoothness[0] = 0

    #计算初始角度的梯度 (只对第一个点有梯度)
    grad_start[1, 0] = 2.0 * (traj[1, 0] - attract_point[0])
    grad_start[1, 1] = 2.0 * (traj[1, 1] - attract_point[1])

    # 组合距离和平滑度的梯度
    grad = grad_distance + lambda_smooth * grad_smoothness + lambda_start * grad_start
    return grad.reshape(-1)

def smoothness_cost(traj):
    diff_x = np.diff(traj[:, 0], 2)
    diff_y = np.diff(traj[:, 1], 2)
    return np.sum(diff_x ** 2 + diff_y ** 2)

def start_angle_cost(traj, attract_point):
    return np.sum((traj[1, 0] - attract_point[0]) ** 2 + (traj[1, 1] - attract_point[1]) ** 2)

def objective_function(traj_flattened, esdf_map, lambda_smooth, lambda_start, x_min, x_max, y_min, y_max, resolution, clearance, attract_point):
    traj = traj_flattened.reshape(-1, 2)
    distance_cost = distance_to_obstacles(traj, esdf_map, x_min, x_max, y_min, y_max, resolution, clearance)
    smoothness_cost_value = smoothness_cost(traj)
    start_angle_cost_value = start_angle_cost(traj, attract_point)
    return distance_cost + lambda_smooth * smoothness_cost_value + lambda_start * start_angle_cost_value

def optimize_trajectory(initial_traj, esdf_map, lambda_smooth, x_min, x_max, y_min, y_max, resolution, clearance = 0.5, start_angle = None):
    lambda_start = 0.0
    attract_point = np.zeros(2)
    initial_traj_flattened = initial_traj.reshape(-1)
    if start_angle != None:
        lambda_start = 4.0 # TODO (make it a parameter)
        length = np.linalg.norm(initial_traj[1] - initial_traj[0])
        start_angle_vector = np.array([np.cos(start_angle), np.sin(start_angle)])
        attract_point = initial_traj[0] +  length * start_angle_vector

    result = minimize(
        objective_function,
        initial_traj_flattened,
        args=(esdf_map, lambda_smooth, lambda_start, x_min, x_max, y_min, y_max, resolution, clearance, attract_point),
        method='L-BFGS-B',
        jac=gradient_objective_function,
    )

    optimized_traj = result.x.reshape(-1, 2)
    return optimized_traj
