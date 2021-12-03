"""Robomimic datasets."""
import json
from typing import List, Tuple

from absl import logging
import cv2
import numpy as np
import h5py

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


def _load_image(path: types.PathType) -> np.ndarray:
  path = gpath.GPath(path)
  with path.open('rb') as f:
    raw_im = np.asarray(bytearray(f.read()), dtype=np.uint8)
    image = cv2.imdecode(raw_im, cv2.IMREAD_COLOR)[:, :, ::-1]  # BGR -> RGB
    image = np.asarray(image).astype(np.float32) / 255.0
    return image


def _load_dataset_ids(data_dir: types.PathType) -> Tuple[List[str], List[str]]:
  """Loads dataset IDs."""
  dataset_json_path = gpath.GPath(data_dir, 'dataset.json')
  logging.info('*** Loading dataset IDs from %s', dataset_json_path)
  with dataset_json_path.open('r') as f:
    dataset_json = json.load(f)
    train_ids = dataset_json['train_ids']
    val_ids = dataset_json['val_ids']

  train_ids = [str(i) for i in train_ids]
  val_ids = [str(i) for i in val_ids]

  return train_ids, val_ids


class RobomimicDataSource(core.DataSource):
  """Data loader for Robomimic datasets."""

  def __init__(
      self,
      data_dir,
      image_scale: int,
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

    # TODO: FIX placeholder
    train_ids, val_ids = ['view0', 'view1'], ['view2', 'view3']
    super().__init__(train_ids=train_ids, val_ids=val_ids,
                     **kwargs)

    # TODO: FIX placeholder
    # self.scene_center, self.scene_scale, self._near, self._far = \
    #   load_scene_info(self.data_dir)
    self.scene_center, self.scene_scale, self._near, self._far = \
      np.array([0., 0., 0.]), 1.0, 0.01, 6.0

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
    metadata_path = self.data_dir / 'metadata.json'
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

  def get_rgb_path(self, item_id):
    return self.rgb_dir / f'{item_id}.png'

  def load_rgb(self, item_id):
    return _load_image(self.rgb_dir / f'{item_id}.png')
    
  def load_camera(self, camera_id, scale_factor=1.0):
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
    return self.metadata_dict[item_id]['appearance_id']

  def get_camera_id(self, item_id):
    return self.metadata_dict[item_id]['camera_id']

  def get_warp_id(self, item_id):
    return self.metadata_dict[item_id]['warp_id']

  def get_time_id(self, item_id):
    if 'time_id' in self.metadata_dict[item_id]:
      return self.metadata_dict[item_id]['time_id']
    else:
      # Fallback for older datasets.
      return self.metadata_dict[item_id]['warp_id']
