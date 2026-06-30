import os
import json
import datetime
import pathlib
import time
import cv2
import carla
from collections import deque
import math
from collections import OrderedDict
import torch
import carla
import numpy as np
from PIL import Image
from torchvision import transforms as T
from Bench2DriveZoo.team_code.pid_controller import PIDController
from Bench2DriveZoo.team_code.planner import RoutePlanner
from leaderboard.autoagents import autonomous_agent
from mmcv import Config
from mmcv.models import build_model
from mmcv.utils import (get_dist_info, init_dist, load_checkpoint,wrap_fp16_model)
from mmcv.datasets.pipelines import Compose
from mmcv.parallel.collate import collate as  mm_collate_to_batch_form
from mmcv.core.bbox import get_box_type
from pyquaternion import Quaternion
from scipy.interpolate import splprep, splev
from scipy.optimize import fsolve
import seaborn as sns
import copy
SAVE_PATH = os.environ.get('SAVE_PATH', None)
IS_BENCH2DRIVE = os.environ.get('IS_BENCH2DRIVE', None)

def get_entry_point():
    return 'UniadAgent'


def float_to_uint8_color(float_clr):
    assert all([c >= 0. for c in float_clr])
    assert all([c <= 1. for c in float_clr])
    return [int(c * 255.) for c in float_clr]


COLORS = [float_to_uint8_color(clr) for clr in sns.color_palette("bright", n_colors=10)]
COLORMAP = OrderedDict({
    6: COLORS[8],  # yellow
    4: COLORS[8],
    3: COLORS[0],  # blue
    1: COLORS[6],  # pink
    0: COLORS[2],  # green
    8: COLORS[7],  # gray
    7: COLORS[1],  # orange
    5: COLORS[3],  # red
    2: COLORS[5],  # brown
})


class UniadAgent(autonomous_agent.AutonomousAgent):
    def setup(self, path_to_conf_file):
        self.track = autonomous_agent.Track.SENSORS
        self.steer_step = 0
        self.last_moving_status = 0
        self.last_moving_step = -1
        self.last_steers = deque()
        self.pidcontroller = PIDController() 
        self.config_path = path_to_conf_file.split('+')[0]
        self.ckpt_path = path_to_conf_file.split('+')[1]
        if IS_BENCH2DRIVE:
            self.save_name = path_to_conf_file.split('+')[-1]
        else:
            self.save_name = '_'.join(map(lambda x: '%02d' % x, (now.month, now.day, now.hour, now.minute, now.second)))
        self.step = -1
        self.wall_start = time.time()
        self.initialized = False
        cfg = Config.fromfile(self.config_path)
        cfg.model['motion_head']['anchor_info_path'] = os.path.join('Bench2DriveZoo',cfg.model['motion_head']['anchor_info_path'])
        if hasattr(cfg, 'plugin'):
            if cfg.plugin:
                import importlib
                if hasattr(cfg, 'plugin_dir'):
                    plugin_dir = cfg.plugin_dir
                    plugin_dir = os.path.join("Bench2DriveZoo", plugin_dir)
                    _module_dir = os.path.dirname(plugin_dir)
                    _module_dir = _module_dir.split('/')
                    _module_path = _module_dir[0]
                    for m in _module_dir[1:]:
                        _module_path = _module_path + '.' + m
                    print(_module_path)
                    plg_lib = importlib.import_module(_module_path)  
  
        self.model = build_model(cfg.model, train_cfg=cfg.get('train_cfg'), test_cfg=cfg.get('test_cfg'))
        checkpoint = load_checkpoint(self.model, self.ckpt_path, map_location='cpu', strict=True)
        self.model.cuda()
        self.model.eval()
        self.inference_only_pipeline = []
        for inference_only_pipeline in cfg.inference_only_pipeline:
            if inference_only_pipeline["type"] not in ['LoadMultiViewImageFromFilesInCeph']:
                self.inference_only_pipeline.append(inference_only_pipeline)
        self.inference_only_pipeline = Compose(self.inference_only_pipeline)
        self.takeover = False
        self.stop_time = 0
        self.takeover_time = 0
        self.save_path = None
        self._im_transform = T.Compose([T.ToTensor(), T.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])])
        self.last_steers = deque()
        self.lat_ref, self.lon_ref = 42.0, 2.0
        control = carla.VehicleControl()
        control.steer = 0.0
        control.throttle = 0.0
        control.brake = 0.0	
        self.prev_control = control
        if SAVE_PATH is not None:
            now = datetime.datetime.now()
            # string = pathlib.Path(os.environ['ROUTES']).stem + '_'
            string = self.save_name
            self.save_path = pathlib.Path(os.environ['SAVE_PATH']) / string
            self.save_path.mkdir(parents=True, exist_ok=False)
            (self.save_path / 'rgb_front').mkdir()
            (self.save_path / 'rgb_front_right').mkdir()
            (self.save_path / 'rgb_front_left').mkdir()
            (self.save_path / 'rgb_back').mkdir()
            (self.save_path / 'rgb_back_right').mkdir()
            (self.save_path / 'rgb_back_left').mkdir()
            (self.save_path / 'meta').mkdir()
            (self.save_path / 'bev').mkdir()
   
        # write extrinsics directly
        self.lidar2img = {
        'CAM_FRONT':np.array([[ 1.14251841e+03,  8.00000000e+02,  0.00000000e+00, -9.52000000e+02],
                                  [ 0.00000000e+00,  4.50000000e+02, -1.14251841e+03, -8.09704417e+02],
                                  [ 0.00000000e+00,  1.00000000e+00,  0.00000000e+00, -1.19000000e+00],
                                 [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  1.00000000e+00]]),
          'CAM_FRONT_LEFT':np.array([[ 6.03961325e-14,  1.39475744e+03,  0.00000000e+00, -9.20539908e+02],
                                   [-3.68618420e+02,  2.58109396e+02, -1.14251841e+03, -6.47296750e+02],
                                   [-8.19152044e-01,  5.73576436e-01,  0.00000000e+00, -8.29094072e-01],
                                   [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  1.00000000e+00]]),
          'CAM_FRONT_RIGHT':np.array([[ 1.31064327e+03, -4.77035138e+02,  0.00000000e+00,-4.06010608e+02],
                                       [ 3.68618420e+02,  2.58109396e+02, -1.14251841e+03,-6.47296750e+02],
                                    [ 8.19152044e-01,  5.73576436e-01,  0.00000000e+00,-8.29094072e-01],
                                    [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00, 1.00000000e+00]]),
         'CAM_BACK':np.array([[-5.60166031e+02, -8.00000000e+02,  0.00000000e+00, -1.28800000e+03],
                     [ 5.51091060e-14, -4.50000000e+02, -5.60166031e+02, -8.58939847e+02],
                     [ 1.22464680e-16, -1.00000000e+00,  0.00000000e+00, -1.61000000e+00],
                     [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  1.00000000e+00]]),
        'CAM_BACK_LEFT':np.array([[-1.14251841e+03,  8.00000000e+02,  0.00000000e+00, -6.84385123e+02],
                                  [-4.22861679e+02, -1.53909064e+02, -1.14251841e+03, -4.96004706e+02],
                                  [-9.39692621e-01, -3.42020143e-01,  0.00000000e+00, -4.92889531e-01],
                                  [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  1.00000000e+00]]),
  
        'CAM_BACK_RIGHT': np.array([[ 3.60989788e+02, -1.34723223e+03,  0.00000000e+00, -1.04238127e+02],
                                    [ 4.22861679e+02, -1.53909064e+02, -1.14251841e+03, -4.96004706e+02],
                                    [ 9.39692621e-01, -3.42020143e-01,  0.00000000e+00, -4.92889531e-01],
                                    [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  1.00000000e+00]])
        }
        self.lidar2cam = {
        'CAM_FRONT':np.array([[ 1.  ,  0.  ,  0.  ,  0.  ],
                                 [ 0.  ,  0.  , -1.  , -0.24],
                                 [ 0.  ,  1.  ,  0.  , -1.19],
                              [ 0.  ,  0.  ,  0.  ,  1.  ]]),
        'CAM_FRONT_LEFT':np.array([[ 0.57357644,  0.81915204,  0.  , -0.22517331],
                                      [ 0.        ,  0.        , -1.  , -0.24      ],
                                   [-0.81915204,  0.57357644,  0.  , -0.82909407],
                                   [ 0.        ,  0.        ,  0.  ,  1.        ]]),
          'CAM_FRONT_RIGHT':np.array([[ 0.57357644, -0.81915204, 0.  ,  0.22517331],
                                   [ 0.        ,  0.        , -1.  , -0.24      ],
                                   [ 0.81915204,  0.57357644,  0.  , -0.82909407],
                                   [ 0.        ,  0.        ,  0.  ,  1.        ]]),
        'CAM_BACK':np.array([[-1. ,  0.,  0.,  0.  ],
                             [ 0. ,  0., -1., -0.24],
                             [ 0. , -1.,  0., -1.61],
                             [ 0. ,  0.,  0.,  1.  ]]),
     
        'CAM_BACK_LEFT':np.array([[-0.34202014,  0.93969262,  0.  , -0.25388956],
                                  [ 0.        ,  0.        , -1.  , -0.24      ],
                                  [-0.93969262, -0.34202014,  0.  , -0.49288953],
                                  [ 0.        ,  0.        ,  0.  ,  1.        ]]),
  
        'CAM_BACK_RIGHT':np.array([[-0.34202014, -0.93969262,  0.  ,  0.25388956],
                                  [ 0.        ,  0.         , -1.  , -0.24      ],
                                  [ 0.93969262, -0.34202014 ,  0.  , -0.49288953],
                                  [ 0.        ,  0.         ,  0.  ,  1.        ]])
        }
        self.lidar2ego = np.array([[ 0. ,  1. ,  0. , -0.39],
                                   [-1. ,  0. ,  0. ,  0.  ],
                                   [ 0. ,  0. ,  1. ,  1.84],
                                   [ 0. ,  0. ,  0. ,  1.  ]])
        
        topdown_extrinsics =  np.array([[0.0, -0.0, -1.0, 50.0], [0.0, 1.0, -0.0, 0.0], [1.0, -0.0, 0.0, -0.0], [0.0, 0.0, 0.0, 1.0]])
        unreal2cam = np.array([[0,1,0,0], [0,0,-1,0], [1,0,0,0], [0,0,0,1]])
        self.coor2topdown = unreal2cam @ topdown_extrinsics
        topdown_intrinsics = np.array([[548.993771650447, 0.0, 256.0, 0], [0.0, 548.993771650447, 256.0, 0], [0.0, 0.0, 1.0, 0], [0, 0, 0, 1.0]])
        self.coor2topdown = topdown_intrinsics @ self.coor2topdown

    def _init(self):
        
        try:
            locx, locy = self._global_plan_world_coord[0][0].location.x, self._global_plan_world_coord[0][0].location.y
            lon, lat = self._global_plan[0][0]['lon'], self._global_plan[0][0]['lat']
            EARTH_RADIUS_EQUA = 6378137.0
            def equations(vars):
                x, y = vars
                eq1 = lon * math.cos(x * math.pi / 180) - (locx * x * 180) / (math.pi * EARTH_RADIUS_EQUA) - math.cos(x * math.pi / 180) * y
                eq2 = math.log(math.tan((lat + 90) * math.pi / 360)) * EARTH_RADIUS_EQUA * math.cos(x * math.pi / 180) + locy - math.cos(x * math.pi / 180) * EARTH_RADIUS_EQUA * math.log(math.tan((90 + x) * math.pi / 360))
                return [eq1, eq2]
            initial_guess = [0, 0]
            solution = fsolve(equations, initial_guess)
            self.lat_ref, self.lon_ref = solution[0], solution[1]
        except Exception as e:
            print(e, flush=True)
            self.lat_ref, self.lon_ref = 0, 0        
        self._route_planner = RoutePlanner(4.0, 50.0, lat_ref=self.lat_ref, lon_ref=self.lon_ref)
        self._route_planner.set_route(self._global_plan, True)
        self.initialized = True
        self.metric_info = {}

    def sensors(self):
        sensors =[
                # camera rgb
                {
                    'type': 'sensor.camera.rgb',
                    'x': 0.80, 'y': 0.0, 'z': 1.60,
                    'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0,
                    'width': 1600, 'height': 900, 'fov': 70,
                    'id': 'CAM_FRONT'
                },
                {
                    'type': 'sensor.camera.rgb',
                    'x': 0.27, 'y': -0.55, 'z': 1.60,
                    'roll': 0.0, 'pitch': 0.0, 'yaw': -55.0,
                    'width': 1600, 'height': 900, 'fov': 70,
                    'id': 'CAM_FRONT_LEFT'
                },
                {
                    'type': 'sensor.camera.rgb',
                    'x': 0.27, 'y': 0.55, 'z': 1.60,
                    'roll': 0.0, 'pitch': 0.0, 'yaw': 55.0,
                    'width': 1600, 'height': 900, 'fov': 70,
                    'id': 'CAM_FRONT_RIGHT'
                },
                {
                    'type': 'sensor.camera.rgb',
                    'x': -2.0, 'y': 0.0, 'z': 1.60,
                    'roll': 0.0, 'pitch': 0.0, 'yaw': 180.0,
                    'width': 1600, 'height': 900, 'fov': 110,
                    'id': 'CAM_BACK'
                },
                {
                    'type': 'sensor.camera.rgb',
                    'x': -0.32, 'y': -0.55, 'z': 1.60,
                    'roll': 0.0, 'pitch': 0.0, 'yaw': -110.0,
                    'width': 1600, 'height': 900, 'fov': 70,
                    'id': 'CAM_BACK_LEFT'
                },
                {
                    'type': 'sensor.camera.rgb',
                    'x': -0.32, 'y': 0.55, 'z': 1.60,
                    'roll': 0.0, 'pitch': 0.0, 'yaw': 110.0,
                    'width': 1600, 'height': 900, 'fov': 70,
                    'id': 'CAM_BACK_RIGHT'
                },
                # imu
                {
                    'type': 'sensor.other.imu',
                    'x': -1.4, 'y': 0.0, 'z': 0.0,
                    'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0,
                    'sensor_tick': 0.05,
                    'id': 'IMU'
                },
                # gps
                {
                    'type': 'sensor.other.gnss',
                    'x': -1.4, 'y': 0.0, 'z': 0.0,
                    'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0,
                    'sensor_tick': 0.01,
                    'id': 'GPS'
                },
                # speed
                {
                    'type': 'sensor.speedometer',
                    'reading_frequency': 20,
                    'id': 'SPEED'
                },
                
            ]
        
        if IS_BENCH2DRIVE:
            sensors += [
                    {	
                        'type': 'sensor.camera.rgb',
                        'x': 0.0, 'y': 0.0, 'z': 50.0,
                        'roll': 0.0, 'pitch': -90.0, 'yaw': 0.0,
                        'width': 512, 'height': 512, 'fov': 5 * 10.0,
                        'id': 'bev'
                    }]
        return sensors

    def tick(self, input_data):
        self.step += 1
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 20]
        imgs = {}
        for cam in ['CAM_FRONT','CAM_FRONT_LEFT','CAM_FRONT_RIGHT','CAM_BACK','CAM_BACK_LEFT','CAM_BACK_RIGHT']:
            img = cv2.cvtColor(input_data[cam][1][:, :, :3], cv2.COLOR_BGR2RGB)
            _, img = cv2.imencode('.jpg', img, encode_param)
            img = cv2.imdecode(img, cv2.IMREAD_COLOR)
            imgs[cam] = img
        bev = cv2.cvtColor(input_data['bev'][1][:, :, :3], cv2.COLOR_BGR2RGB)
        gps = input_data['GPS'][1][:2]
        speed = input_data['SPEED'][1]['speed']
        compass = input_data['IMU'][1][-1]
        acceleration = input_data['IMU'][1][:3]
        angular_velocity = input_data['IMU'][1][3:6]
        pos = self.gps_to_location(gps)
        near_node, near_command = self._route_planner.run_step(pos)
        if (math.isnan(compass) == True): #It can happen that the compass sends nan for a few frames
            compass = 0.0
            acceleration = np.zeros(3)
            angular_velocity = np.zeros(3)

        result = {
                'imgs': imgs,
                'gps': gps,
                'pos':pos,
                'speed': speed,
                'compass': compass,
                'bev': bev,
                'acceleration':acceleration,
                'angular_velocity':angular_velocity,
                'command_near':near_command,
                'command_near_xy':near_node
    
                }
        
        return result
    
    @torch.no_grad()
    def run_step(self, input_data, timestamp):
        if not self.initialized:
            self._init()
        tick_data = self.tick(input_data)
        results = {}
        results['lidar2img'] = []
        results['lidar2cam'] = []
        results['img'] = []
        results['folder'] = ' '
        results['scene_token'] = ' '  
        results['frame_idx'] = 0
        results['timestamp'] = self.step / 20
        results['box_type_3d'], _ = get_box_type('LiDAR')
  
        for cam in ['CAM_FRONT','CAM_FRONT_LEFT','CAM_FRONT_RIGHT','CAM_BACK','CAM_BACK_LEFT','CAM_BACK_RIGHT']:
            results['lidar2img'].append(self.lidar2img[cam])
            results['lidar2cam'].append(self.lidar2cam[cam])
            results['img'].append(tick_data['imgs'][cam])
        results['lidar2img'] = np.stack(results['lidar2img'],axis=0)
        results['lidar2cam'] = np.stack(results['lidar2cam'],axis=0)
  
        raw_theta = tick_data['compass']   if not np.isnan(tick_data['compass']) else 0
        ego_theta = -raw_theta + np.pi/2
        rotation = list(Quaternion(axis=[0, 0, 1], radians=ego_theta))
  
        can_bus = np.zeros(18)
        can_bus[0] = tick_data['pos'][0]
        can_bus[1] = -tick_data['pos'][1]
        can_bus[3:7] = rotation
        can_bus[7] = tick_data['speed']
        can_bus[10:13] = tick_data['acceleration']
        can_bus[11] *= -1
        can_bus[13:16] = -tick_data['angular_velocity']
        can_bus[16] = ego_theta
        can_bus[17] = ego_theta / np.pi * 180 
        results['can_bus'] = can_bus
        command = tick_data['command_near']
        if command < 0:
            command = 4
        command -= 1
        results['command'] = command
  
        theta_to_lidar = raw_theta
        command_near_xy = np.array([tick_data['command_near_xy'][0]-can_bus[0],-tick_data['command_near_xy'][1]-can_bus[1]])
        rotation_matrix = np.array([[np.cos(theta_to_lidar),-np.sin(theta_to_lidar)],[np.sin(theta_to_lidar),np.cos(theta_to_lidar)]])
        local_command_xy = rotation_matrix @ command_near_xy
  
        ego2world = np.eye(4)
        ego2world[0:3,0:3] = Quaternion(axis=[0, 0, 1], radians=ego_theta).rotation_matrix
        ego2world[0:2,3] = can_bus[0:2]
        lidar2global = ego2world @ self.lidar2ego
        results['l2g_r_mat'] = lidar2global[0:3,0:3]
        results['l2g_t'] = lidar2global[0:3,3]
        stacked_imgs = np.stack(results['img'],axis=-1)
        results['img_shape'] = stacked_imgs.shape
        results['ori_shape'] = stacked_imgs.shape
        results['pad_shape'] = stacked_imgs.shape
        results = self.inference_only_pipeline(results)
        self.device="cuda"
        input_data_batch = mm_collate_to_batch_form([results], samples_per_gpu=1)
        for key, data in input_data_batch.items():
            if key != 'img_metas':
                if torch.is_tensor(data[0]):
                    data[0] = data[0].to(self.device)
        output_data_batch = self.model(input_data_batch, return_loss=False, rescale=True)
        self._current_output = output_data_batch  # store for visualization
        out_truck =  output_data_batch[0]['planning']['result_planning']['sdc_traj'][0].cpu().numpy()
        steer_traj, throttle_traj, brake_traj, metadata_traj = self.pidcontroller.control_pid(out_truck, tick_data['speed'], local_command_xy)
        if brake_traj < 0.05: brake_traj = 0.0
        if throttle_traj > brake_traj: brake_traj = 0.0
        if tick_data['speed']>5:
            throttle_traj = 0
        control = carla.VehicleControl()
        self.pid_metadata = metadata_traj
        self.pid_metadata['agent'] = 'only_traj'
        control.steer = np.clip(float(steer_traj), -1, 1)
        control.throttle = np.clip(float(throttle_traj), 0, 0.75)
        control.brake = np.clip(float(brake_traj), 0, 1)
        self.pid_metadata['steer'] = control.steer
        self.pid_metadata['throttle'] = control.throttle
        self.pid_metadata['brake'] = control.brake
        self.pid_metadata['steer_traj'] = float(steer_traj)
        self.pid_metadata['throttle_traj'] = float(throttle_traj)
        self.pid_metadata['brake_traj'] = float(brake_traj)
        self.pid_metadata['plan'] = out_truck.tolist()
        metric_info = self.get_metric_info()
        self.metric_info[self.step] = metric_info
        if SAVE_PATH is not None and self.step % 1 == 0:
            self.save(tick_data, out_truck)
        self.prev_control = control
        return control

    def save(self, tick_data, ego_traj=None):
        frame = self.step  # save every frame (consistent with VAD visualize)
        output = self._current_output[0] if hasattr(self, '_current_output') else None

        imgs_with_box = {}
        for cam in ['CAM_FRONT','CAM_FRONT_LEFT','CAM_FRONT_RIGHT','CAM_BACK','CAM_BACK_LEFT','CAM_BACK_RIGHT']:
            img = tick_data['imgs'][cam]
            # Try to draw 3D detection/tracking boxes
            try:
                if output is not None and 'pts_bbox' in output:
                    img = self._draw_boxes_on_img(output['pts_bbox'], img,
                                                   self.lidar2img[cam],
                                                   canvas_size=(900, 1600))
                elif output is not None and 'track_results' in output:
                    img = self._draw_track_results(output, img,
                                                    self.lidar2img[cam],
                                                    canvas_size=(900, 1600))
            except Exception:
                pass  # skip visualization errors, keep raw image
            imgs_with_box[cam] = img

        # BEV image with annotations
        bev_img = tick_data['bev'].copy()
        try:
            if output is not None and 'pts_bbox' in output:
                bev_img = self._draw_boxes_on_img(output['pts_bbox'], bev_img,
                                                   self.coor2topdown,
                                                   canvas_size=(512, 512))
            # Draw ego trajectory on BEV
            if ego_traj is not None:
                bev_img = self._draw_traj_bev(ego_traj, bev_img, is_ego=True)
        except Exception:
            pass

        # Draw ego trajectory on front camera
        if ego_traj is not None:
            try:
                imgs_with_box['CAM_FRONT'] = self._draw_traj(ego_traj,
                                                              imgs_with_box['CAM_FRONT'])
            except Exception:
                pass

        # Save all annotated images
        for cam, img in imgs_with_box.items():
            cam_dir = 'rgb_' + cam.lower().replace('cam_', '')
            Image.fromarray(img).save(self.save_path / cam_dir / ('%04d.png' % frame))
        Image.fromarray(bev_img).save(self.save_path / 'bev' / ('%04d.png' % frame))

        # Save metadata
        outfile = open(self.save_path / 'meta' / ('%04d.json' % frame), 'w')
        json.dump(self.pid_metadata, outfile, indent=4)
        outfile.close()

        # metric info
        outfile = open(self.save_path / 'metric_info.json', 'w')
        json.dump(self.metric_info, outfile, indent=4)
        outfile.close()

    # ── Drawing helper methods ──

    def _draw_traj(self, traj, raw_img, canvas_size=(900, 1600), thickness=3,
                    is_ego=True, hue_start=120, hue_end=80):
        """Draw ego trajectory on front camera image using lidar2img projection."""
        line = traj
        lidar2img_rt = self.lidar2img['CAM_FRONT']
        img = raw_img.copy()
        pts_4d = np.stack([line[:, 0], line[:, 1],
                           np.ones((line.shape[0])) * (-1.84),
                           np.ones((line.shape[0]))])
        pts_2d = ((lidar2img_rt @ pts_4d).T)
        pts_2d[:, 0] /= pts_2d[:, 2]
        pts_2d[:, 1] /= pts_2d[:, 2]
        mask = ((pts_2d[:, 0] > 0) & (pts_2d[:, 0] < canvas_size[1]) &
                (pts_2d[:, 1] > 0) & (pts_2d[:, 1] < canvas_size[0]))
        if not mask.any():
            return img
        pts_2d = pts_2d[mask, 0:2]
        if is_ego:
            pts_2d = np.concatenate([np.array([[800, 900]]), pts_2d], axis=0)
        try:
            tck, u = splprep([pts_2d[:, 0], pts_2d[:, 1]], s=0)
        except Exception:
            return img
        unew = np.linspace(0, 1, 100)
        smoothed_pts = np.stack(splev(unew, tck)).astype(int).T
        num_points = len(smoothed_pts)
        for i in range(num_points - 1):
            hue = hue_start + (hue_end - hue_start) * (i / num_points)
            hsv_color = np.array([hue, 255, 255], dtype=np.uint8)
            rgb_color = cv2.cvtColor(hsv_color[np.newaxis, np.newaxis, :],
                                      cv2.COLOR_HSV2RGB).reshape(-1)
            rgb_color_tuple = (float(rgb_color[0]), float(rgb_color[1]),
                                float(rgb_color[2]))
            cv2.line(img, (smoothed_pts[i, 0], smoothed_pts[i, 1]),
                     (smoothed_pts[i + 1, 0], smoothed_pts[i + 1, 1]),
                     color=rgb_color_tuple, thickness=thickness)
        return img

    def _draw_traj_bev(self, traj, raw_img, canvas_size=(512, 512), thickness=3,
                        is_ego=True, hue_start=120, hue_end=80):
        """Draw ego trajectory on BEV image."""
        if is_ego:
            line = np.concatenate([np.zeros((1, 2)), traj], axis=0)
        else:
            line = traj
        img = raw_img.copy()
        pts_4d = np.stack([line[:, 0], line[:, 1],
                           np.zeros((line.shape[0])),
                           np.ones((line.shape[0]))])
        pts_2d = (self.coor2topdown @ pts_4d).T
        pts_2d[:, 0] /= pts_2d[:, 2]
        pts_2d[:, 1] /= pts_2d[:, 2]
        mask = ((pts_2d[:, 0] > 0) & (pts_2d[:, 0] < canvas_size[1]) &
                (pts_2d[:, 1] > 0) & (pts_2d[:, 1] < canvas_size[0]))
        if not mask.any():
            return img
        pts_2d = pts_2d[mask, 0:2]
        try:
            tck, u = splprep([pts_2d[:, 0], pts_2d[:, 1]], s=0)
        except Exception:
            return img
        unew = np.linspace(0, 1, 100)
        smoothed_pts = np.stack(splev(unew, tck)).astype(int).T
        num_points = len(smoothed_pts)
        for i in range(num_points - 1):
            hue = hue_start + (hue_end - hue_start) * (i / num_points)
            hsv_color = np.array([hue, 255, 255], dtype=np.uint8)
            rgb_color = cv2.cvtColor(hsv_color[np.newaxis, np.newaxis, :],
                                      cv2.COLOR_HSV2RGB).reshape(-1)
            rgb_color_tuple = (float(rgb_color[0]), float(rgb_color[1]),
                                float(rgb_color[2]))
            if (smoothed_pts[i, 0] > 0 and smoothed_pts[i, 0] < canvas_size[1] and
                smoothed_pts[i, 1] > 0 and smoothed_pts[i, 1] < canvas_size[0]):
                cv2.line(img, (smoothed_pts[i, 0], smoothed_pts[i, 1]),
                         (smoothed_pts[i + 1, 0], smoothed_pts[i + 1, 1]),
                         color=rgb_color_tuple, thickness=thickness)
            elif i == 0:
                break
        return img

    def _draw_boxes_on_img(self, pts_bbox, raw_img, lidar2img_rt,
                            canvas_size=(900, 1600), thickness=1):
        """Draw 3D bboxes from model output on image or BEV."""
        bboxes3d = pts_bbox.get('boxes_3d', None)
        if bboxes3d is None:
            return raw_img
        scores = pts_bbox.get('scores_3d', None)
        labels = pts_bbox.get('labels_3d', None)
        trajs = pts_bbox.get('trajs_3d', None)

        img = raw_img.copy()
        bboxes3d_numpy = bboxes3d.tensor.cpu().numpy()
        if len(bboxes3d_numpy) == 0:
            return img

        corners_3d = bboxes3d.corners
        num_bbox = corners_3d.shape[0]
        pts_4d = np.concatenate(
            [corners_3d.reshape(-1, 3),
             np.ones((num_bbox * 8, 1))], axis=-1)
        lidar2img_rt = copy.deepcopy(lidar2img_rt).reshape(4, 4)
        if isinstance(lidar2img_rt, torch.Tensor):
            lidar2img_rt = lidar2img_rt.cpu().numpy()
        if isinstance(scores, torch.Tensor):
            scores = scores.cpu().numpy()
        if isinstance(labels, torch.Tensor):
            labels = labels.cpu().numpy()

        pts_2d = (lidar2img_rt @ pts_4d.T).T
        pts_2d[:, 0] /= pts_2d[:, 2]
        pts_2d[:, 1] /= pts_2d[:, 2]
        imgfov_pts_2d = pts_2d[..., :2].reshape(num_bbox, 8, 2)
        depth = pts_2d[..., 2].reshape(num_bbox, 8)
        mask1 = ((imgfov_pts_2d[:, :, 0] > -1e5) &
                 (imgfov_pts_2d[:, :, 0] < 1e5) &
                 (imgfov_pts_2d[:, :, 1] > -1e5) &
                 (imgfov_pts_2d[:, :, 1] < 1e5) &
                 (depth > -1)).all(-1)
        mask2 = ((imgfov_pts_2d.reshape(num_bbox, 16).max(axis=-1) -
                  imgfov_pts_2d.reshape(num_bbox, 16).min(axis=-1)) < 2000)
        mask = mask1 & mask2
        if scores is not None:
            mask3 = (scores >= 0.3)
            mask = mask & mask3
        if not mask.any():
            return img

        scores = scores[mask] if scores is not None else None
        labels = labels[mask] if labels is not None else None
        imgfov_pts_2d = imgfov_pts_2d[mask]
        num_bbox = mask.sum()

        # Draw agent future trajectories on BEV
        if trajs is not None:
            trajs = trajs[mask]
            agent_boxes = bboxes3d_numpy[mask]
            for traj, agent_box, label in zip(trajs, agent_boxes, labels):
                if label in [0, 1, 2, 3, 7]:
                    for i in range(min(6, traj.shape[0])):
                        traj1 = np.concatenate(
                            [np.zeros((1, 2)), traj[i].reshape(6, 2)], axis=0)
                        traj1 = np.cumsum(traj1, axis=0) + agent_box[None, 0:2]
                        if canvas_size == (900, 1600):
                            img = self._draw_traj(traj1, img,
                                                   hue_start=0, hue_end=20)
                        else:
                            img = self._draw_traj_bev(traj1, img,
                                                       hue_start=0, hue_end=20)

        return self._plot_rect3d_on_img(img, num_bbox, imgfov_pts_2d,
                                         scores, labels, thickness,
                                         bev=(canvas_size != (900, 1600)))

    def _draw_track_results(self, output, raw_img, lidar2img_rt,
                             canvas_size=(900, 1600)):
        """Draw boxes from UniAD tracking results (alternative output format)."""
        # UniAD may output track results differently than VAD-style pts_bbox
        if 'track_boxes' in output:
            # Try to access tracking output - format depends on UniAD version
            pass
        return raw_img

    def _plot_rect3d_on_img(self, img, num_rects, rect_corners,
                              scores=None, labels=None, thickness=1,
                              bev=False):
        """Draw 3D bounding box edges on image."""
        line_indices = ((0, 1), (0, 3), (0, 4), (1, 2), (1, 5),
                        (3, 2), (3, 7), (4, 5), (4, 7), (2, 6), (5, 6), (6, 7))
        if bev:
            line_indices = ((0, 3), (3, 7), (4, 7), (0, 4))
        for i in range(num_rects):
            label = labels[i] if labels is not None else 0
            c = COLORMAP.get(label, COLORS[0])
            corners = rect_corners[i].astype(np.int32)
            if scores is not None and scores[i] < 0.3:
                continue
            for start, end in line_indices:
                cv2.line(img, (corners[start, 0], corners[start, 1]),
                         (corners[end, 0], corners[end, 1]), c,
                         thickness, cv2.LINE_AA)
        return img.astype(np.uint8)

    def destroy(self):
        del self.model
        torch.cuda.empty_cache()

    def gps_to_location(self, gps):
        EARTH_RADIUS_EQUA = 6378137.0
        # gps content: numpy array: [lat, lon, alt]
        lat, lon = gps
        scale = math.cos(self.lat_ref * math.pi / 180.0)
        my = math.log(math.tan((lat+90) * math.pi / 360.0)) * (EARTH_RADIUS_EQUA * scale)
        mx = (lon * (math.pi * EARTH_RADIUS_EQUA * scale)) / 180.0
        y = scale * EARTH_RADIUS_EQUA * math.log(math.tan((90.0 + self.lat_ref) * math.pi / 360.0)) - my
        x = mx - scale * self.lon_ref * math.pi * EARTH_RADIUS_EQUA / 180.0
        return np.array([x, y])
