
import argparse
import omni
import os
import json
parser = argparse.ArgumentParser()
from isaacsim import SimulationApp
parser_gpu = argparse.ArgumentParser(description='Specify the GPU to be used and the file to be processed.')
parser_gpu.add_argument('--gpu', type=str, default='0', help='GPU to be used')


args_gpu = parser_gpu.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = args_gpu.gpu
# os.environ['CUDA_VISIBLE_DEVICES'] = '0, 1'

simulation_app = SimulationApp({"headless": True})

from pxr import Usd, UsdGeom,UsdLux
import numpy as np
import open3d as o3d

import sys
import omni
from urllib.parse import quote
from usd_utils import remove_empty_prims, recursive_parse_new, get_mesh_from_points_and_faces, sample_points_from_mesh, sample_points_from_prim
from usd_utils import fix_mdls
from usd_utils import IsEmpty, IsObjXform
from usd_utils import filter_free_noise

from omni.isaac.sensor import Camera
from omni.isaac.core import World

from omni.isaac.core.utils.stage import open_stage
from omni.isaac.core.utils.stage import add_reference_to_stage
import omni.isaac.core.utils.stage as stage_utils
import omni.isaac.core.utils.numpy.rotations as rot_utils
from omni.isaac.core.utils.prims import delete_prim, create_prim
import imageio
from path_utils.advanced_planner import PathPlanner
from path_utils.geometry_tools import *


# setting 

# turn on the camera light
# action_registry = omni.kit.actions.core.get_action_registry()
# action = action_registry.get_action("omni.kit.viewport.menubar.lighting", "set_lighting_mode_camera")
# action.execute()

# AE
omni.kit.commands.execute('ChangeSetting', path='/rtx/post/histogram/enabled', value=True)
omni.kit.commands.execute('ChangeSetting', path='/rtx/post/histogram/whiteScale', value=3.5)

# Adjust the lighting in the original file
def enumerate_lights(stage):
    light_types = [
        "DistantLight",
        "SphereLight",
        "DiskLight",
        "RectLight",
        "CylinderLight"
    ]

    for prim in stage.Traverse():
        prim_type_name = prim.GetTypeName()
        if prim_type_name in light_types:
            UsdGeom.Imageable(prim).MakeVisible()
    #return lights



def convert_usd_to_points(stage, meters_per_unit):
    remove_empty_prims(stage)
    world_prim = stage.GetPrimAtPath("/World")
    prims_all = [p for p in world_prim.GetAllChildren() if p.IsA(UsdGeom.Mesh) or p.IsA(UsdGeom.Xform) and not IsEmpty(p) and IsObjXform(p)]
    pcs_all = []
    sample_points_number = 100000
    for prim in prims_all:
        pcs, mesh = sample_points_from_prim(prim, sample_points_number)
        pcs_all.append(pcs)
    pcs_all = np.concatenate(pcs_all, axis=0) * meters_per_unit
    scene_pcd = o3d.geometry.PointCloud()
    scene_pcd.points = o3d.utility.Vector3dVector(pcs_all)
    scene_pcd.colors = o3d.utility.Vector3dVector(np.ones_like(pcs_all)*0.4)
    scene_pcd = scene_pcd.voxel_down_sample(0.05)
    return scene_pcd


def find_usd_path(dir):
    paths = os.listdir(dir)
    for p in paths:
        if ".usd" in p:
            usd_path = os.path.join(dir,p)
            encoded_uri = quote(usd_path, safe=':/')
            return encoded_uri

parser = argparse.ArgumentParser()
parser.add_argument("--scene_index",type=int,default=4)
parser.add_argument("--gpu_id",type=int,default=0)
parser.add_argument("--grutopia", help="Path to the grutopia file", default='/home/caopeizhou/projects/NavDataGenerator/GRGenerator/data/part2/110_usd/')
parser.add_argument("--default_mdl", help="Path to the mdl file", default='/home/caopeizhou/projects/NavDataGenerator/GRGenerator/mdls/default.mdl')
parser.add_argument("--output_dir", help="Path to where the data should be saved",default="/home/caopeizhou/projects/NavDataGenerator/GRGenerator/output/grutopia_v2/")
parser.add_argument("--scene_scale",type=float,default=1000)
parser.add_argument("--image_height",type=int,default=180)
parser.add_argument("--image_width",type=int,default=320)
parser.add_argument("--camera_hfov",type=float,default=68)
parser.add_argument("--camera_vfov",type=float,default=42)
parser.add_argument("--ceiling_height",type=float,default=1.8)
parser.add_argument("--safe_distance",type=float,default=0.25)
args = parser.parse_known_args()[0]
if not os.path.exists(args.grutopia):
    raise Exception("One of the two folders does not exist!")

# house_id = sorted(os.listdir(args.grutopia))[args.scene_index]


# Loop test
for house_id in sorted(os.listdir(args.grutopia)):
    # if house_id == '0001' or house_id == '0009':
    #     continue
    # if house_id != '0009':
    #     continue
    print(house_id)
    if not os.path.exists("%s/%s"%(args.output_dir,house_id)):
        usd_path = find_usd_path(os.path.join(args.grutopia,house_id))
        print(usd_path)
        world = World(physics_dt=0.01, rendering_dt=0.01, stage_units_in_meters=1.0)
        fix_mdls(usd_path, args.default_mdl)
        add_reference_to_stage(usd_path, "/World/scene")
        stage = omni.usd.get_context().get_stage()
        meters_per_unit = Usd.Stage.Open(usd_path).GetMetadata('metersPerUnit')
        print(meters_per_unit)
        enumerate_lights(stage)
        #set_lights_visible(lights)
        
        # import pdb
        # pdb.set_trace()
        scene_pcd = convert_usd_to_points(stage, meters_per_unit)
        scene_pcd = filter_free_noise(scene_pcd)
        points = np.array(scene_pcd.points)
        floor_heights = extract_floor_heights(points)
        path_planner = PathPlanner(ceiling_offset=args.ceiling_height,safe_distance=args.safe_distance)
        world = World(physics_dt=0.01, rendering_dt=0.01, stage_units_in_meters=1.0)
        # initialize the camera function
        camera = Camera(
            prim_path="/World/camera",
            position=np.array([0,0,0]),
            dt=0.05,
            resolution=(1280, 720),
            orientation=rot_utils.euler_angles_to_quats(np.array([0.0, 0.0, 90.0]),degrees=True))
        camera.set_focal_length(1.4)
        camera.set_focus_distance(0.205)
        camera.set_clipping_range(0.01,10000000)
        world.reset()
        camera.initialize()
        camera.add_motion_vectors_to_frame()
        camera.add_distance_to_image_plane_to_frame()
        for i in range(30):
            world.step(render=True)
        
        for floor_index,current_floor in enumerate(floor_heights):
            current_floor = current_floor[0]
            try:
                path_planner.reset(current_floor,0.5,scene_pcd)
            except:
                continue
            
            for trajectory_num in range(20):
                print(trajectory_num)
                trajectory_index = floor_index * 100 + trajectory_num
                camera_height = np.random.uniform(0.25,1.25)
                camera_intrinsic = generate_intrinsic(args.image_width,args.image_height,args.camera_hfov,args.camera_vfov)
                random_index = np.random.choice(np.nonzero(path_planner.safe_value > 0.1)[0])
                camera_translation = np.array(path_planner.navigable_pcd.points)[random_index]
                camera_translation[2] = camera_height + current_floor
                pitch_rad = np.deg2rad((210 - 180*camera_height))
                camera_rotation = [np.clip(pitch_rad,np.pi/3,np.pi/2),0,0]
                
                tobase_extrinsic = build_transformation_mat(camera_translation * np.array([0,0,1]),camera_rotation)
                distance = np.sum(np.abs(np.array(path_planner.navigable_pcd.points) - camera_translation),axis=-1)
                condition = np.where((path_planner.safe_value>0.2) & (distance > 5.0))[0]
                if condition.shape[0] == 0:
                    print("condition shape0 is 0")
                    continue
                target_point = np.array(path_planner.navigable_pcd.points)[np.random.choice(condition)]
                status,waypoints,wayrotations = path_planner.generate_trajectory(camera_translation,target_point)
                if status == False:
                    print(status)
                    continue
                waypoints = np.concatenate((waypoints,np.ones((waypoints.shape[0],1)) * camera_translation[2]),axis=-1)
                wayrotations = np.stack((np.array([camera_rotation[0]]*waypoints.shape[0]),np.zeros((waypoints.shape[0],)),wayrotations),axis=-1)
                path_pcd = cpu_pointcloud_from_array(waypoints,np.ones_like(waypoints) * np.array([0,0,0]))
                if waypoints.shape[0] > 500:
                    print(waypoints.shape[0])
                    continue
                os.makedirs("%s/%s/trajectory_%d/"%(args.output_dir,house_id,trajectory_index),exist_ok=False)
                os.makedirs("%s/%s/trajectory_%d/rgb/"%(args.output_dir,house_id,trajectory_index),exist_ok=False)
                os.makedirs("%s/%s/trajectory_%d/depth/"%(args.output_dir,house_id,trajectory_index),exist_ok=False)
                o3d.io.write_point_cloud("%s/%s/trajectory_%d/path.ply"%(args.output_dir,house_id,trajectory_index),path_pcd+path_planner.navigable_pcd)
                cv2.imwrite("%s/%s/trajectory_%d/decision_map.jpg"%(args.output_dir,house_id,trajectory_index),path_planner.color_decision_map)
                fps_writer = imageio.get_writer("%s/%s/trajectory_%d/fps.mp4"%(args.output_dir,house_id,trajectory_index), fps=10)
                depth_writer = imageio.get_writer("%s/%s/trajectory_%d/depth.mp4"%(args.output_dir,house_id,trajectory_index), fps=10)
                camera_trajectory = []
                for pt,rt in zip(waypoints,wayrotations):
                    camera_trajectory.append(build_transformation_mat(pt, rt).tolist())
                camera_trajectory = np.array(camera_trajectory)
                
                for frame_index,camera_ext in enumerate(camera_trajectory):
                    camera_pos = camera_ext[0:3,3]/meters_per_unit
                    camera_rot = camera_ext[0:3,0:3]
                    camera_euler_angles = rot_utils.rot_matrices_to_quats(camera_rot)
                    camera_euler_angles = rot_utils.quats_to_euler_angles(camera_euler_angles)
                    camera_euler_angles[0],camera_euler_angles[1],camera_euler_angles[2] = camera_euler_angles[1],np.clip((np.pi/2 - camera_euler_angles[0])/2.0,0,np.pi/8),camera_euler_angles[2]+np.pi/2
                    camera.set_world_pose(camera_pos,rot_utils.euler_angles_to_quats(camera_euler_angles))
                    for i in range(8):
                        world.step(render=True)
                    
                    # import pdb
                    # pdb.set_trace()
                    
                    rgb = cv2.cvtColor(camera.get_rgba()[:,:,:3],cv2.COLOR_BGR2RGB)
                    depth = camera.get_depth()
                    depth[np.isinf(depth)] = 0.0
                    depth = depth * meters_per_unit
                    fps_writer.append_data(camera.get_rgba()[:,:,:3])
                    depth_writer.append_data((np.clip(depth/5.0,0,1)*255.0).astype(np.uint16))
                    cv2.imwrite(os.path.join(args.output_dir,house_id,"trajectory_%d/"%trajectory_index,'rgb/',"%d.jpg"%frame_index),rgb)
                    cv2.imwrite(os.path.join(args.output_dir,house_id,"trajectory_%d/"%trajectory_index,'depth/',"%d.png"%frame_index),(np.clip(depth*10000.0,0,65535)).astype(np.uint16))

                fps_writer.close()
                depth_writer.close()
                save_dict = {'camera_intrinsic':camera_intrinsic.tolist(),
                            'camera_extrinsic':tobase_extrinsic.tolist(),
                            'camera_trajectory':camera_trajectory.tolist()}
                json_object = json.dumps(save_dict, indent=4)
                with open("%s/%s/trajectory_%d/data.json"%(args.output_dir,house_id,trajectory_index), "w") as outfile:
                    outfile.write(json_object)

        delete_prim("/World/scene")

                



