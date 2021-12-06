"""Robomimic datasets."""
import json
from typing import List, Tuple

from absl import logging
import cv2
import numpy as np
import h5py
import re
from sklearn.model_selection import train_test_split

from nerfies import gpath
from nerfies import types
from nerfies import utils
from nerfies import camera as cam
from nerfies.datasets import core


def load_scene_info(
    data_dir: types.PathType) -> Tuple[np.ndarray, float, float, float]:
  """Loads the scene scale from scene_scale.npy.

  Args:
    data_dir: the path to the dataset.

  Returns:
    scene_center: the center of the scene (unscaled coordinates).
    scene_scale: the scale of the scene.
    near: the near plane of the scene (scaled coordinates).
    far: the far plane of the scene (scaled coordinates).

  Raises:
    ValueError if scene_scale.npy does not exist.
  """
  scene_json_path = gpath.GPath(data_dir, 'scene.json')
  with scene_json_path.open('r') as f:
    scene_json = json.load(f)

  scene_center = np.array(scene_json['center'])
  scene_scale = scene_json['scale']
  near = scene_json['near']
  far = scene_json['far']

  return scene_center, scene_scale, near, far

class RobomimicDataSource(core.DataSource):
  """Data loader for Robomimic datasets."""

  def __init__(
      self,
      data_dir,
      image_scale: int = 1,
      camera_height: int = 84,
      camera_width: int = 84,
      camera_fovy: float = 45.,
      **kwargs):
    self.data_dir = gpath.GPath(data_dir)
    hdf5_path = gpath.GPath(data_dir, 'multiview.hdf5')
    self.data_hdf5 = h5py.File(hdf5_path, "r")

    # Load IDs from JSON if it exists. This is useful since COLMAP fails on
    # some images so this gives us the ability to skip invalid images.
    # train_ids, val_ids = _load_dataset_ids(self.data_dir)

    train_ids, val_ids = self.load_dataset_ids()
    super().__init__(train_ids=train_ids, val_ids=val_ids,
                     **kwargs)

    # TODO: FIX placeholder
    # self.scene_center, self.scene_scale, self._near, self._far = \
    #   load_scene_info(self.data_dir)
    self.scene_center, self.scene_scale, self._near, self._far = \
      np.array([0., 0., 1.0]), 1.0, 0.01, 6.0

    self.image_scale = image_scale

    # TODO: FIX here define here for now 
    self.camera_height = camera_height
    self.camera_width = camera_width
    self.camera_fovy = camera_fovy

    camera_json_path = gpath.GPath(data_dir, 'multiview.json')
    with camera_json_path.open('r') as fp:
      camera_json = json.load(fp)
    self.camera_configs = {}
    for config in camera_json:
      name = config['camera_name']
      config.update(
        {
          'height': self.camera_height,
          'width': self.camera_width,
          'fovy': self.camera_fovy,
        }
      )
      self.camera_configs[name] = config
    metadata_path = self.data_dir / 'multiview_metadata.json'
    self.metadata_dict = None
    if metadata_path.exists():
      with metadata_path.open('r') as f:
        self.metadata_dict = json.load(f)

  @property
  def near(self):
    return self._near

  @property
  def far(self):
    return self._far
  
  def load_dataset_ids(self, seed=1234):
    demo_spec = {}
    view_regrex = r'view(\d+)_image'
    for key in self.data_hdf5['data'].keys():
      demo_spec[key] = {
        # assume the number of frames and views is the same for all views within a demo
        "num_frames": self.data_hdf5['data'][key].attrs['num_samples'],
        "views": [view_name for view_name in self.data_hdf5['data'][key]['obs'].keys() if re.match(view_regrex, view_name)]
      }

    dataset_ids = []
    global_frame_id = 0
    for demo_name, demo in demo_spec.items():
      for frame_id in range(demo['num_frames']):
        for view in demo['views']:
          dataset_ids.append({
            'demo': demo_name,
            'view': view,
            'frame': frame_id,
            'global_frame_id': global_frame_id,
          })
        global_frame_id += 1
    train_ids, val_ids = train_test_split(dataset_ids, test_size=0.2, random_state=seed)
    return train_ids, val_ids

  def load_rgb(self, item_id: dict) -> np.ndarray:
    demo_id = item_id['demo']  # demo_X
    view_id = item_id['view']  # viewX_image
    frame_id = item_id['frame']
    image = self.data_hdf5['data/{}/obs/{}'.format(demo_id, view_id)][frame_id]
    image = image.astype(np.float32) / 255.0
    return image
  
  def load_camera(self, item_id, scale_factor=1.0):
    camera_id = re.sub('(.*)_image', r'\1', item_id['view'])  # viewX_image -> viewX
    return core.load_mujoco_camera(
      camera_config = self.camera_configs[camera_id],
      scale_factor=scale_factor / self.image_scale,
      scene_center=self.scene_center,
      scene_scale=self.scene_scale
    )
    
  def load_test_cameras(self, count=None):
    raise NotImplementedError()

  def load_points(self, shuffle=False):
    raise NotImplementedError()

  def get_appearance_id(self, item_id):
    return item_id['global_frame_id']

  def get_camera_id(self, item_id):
    return int(re.sub('view(.*)_image', r'\1', item_id['view']))
  
  def get_warp_id(self, item_id):
    return item_id['global_frame_id']

  def get_time_id(self, item_id):
    return item_id['global_frame_id']
