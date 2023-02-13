# Copyright (c) OpenMMLab. All rights reserved.
import copy
import os
import os.path as osp
import warnings
from argparse import ArgumentParser
from datetime import datetime
from typing import Optional, Tuple

import cv2
import mmcv
import numba
import numpy as np
import plotly.express as px

from constants import *
from mmpose.apis import (
  collect_multi_frames,
  extract_pose_sequence,
  get_track_id,
  inference_pose_lifter_model,
  inference_top_down_pose_model,
  init_pose_model,
  process_mmdet_results,
  vis_3d_pose_result,
)
from mmpose.core import Smoother
from mmpose.datasets import DatasetInfo
from mmpose.models import PoseLifter, TopDown
from utils import occlude_video_region, subclip_video

debug = False

if debug:
  print("Importing mmdet...")
try:
  from mmdet.apis import inference_detector, init_detector

  has_mmdet = True
  if debug:
    print("success")
except (ImportError, ModuleNotFoundError):
  if debug:
    print("failed")
  has_mmdet = False


@numba.jit()
def convert_keypoint_definition(keypoints, pose_det_dataset, pose_lift_dataset):
  """Convert pose det dataset keypoints definition to pose lifter dataset
  keypoints definition, so that they are compatible with the definitions
  required for 3D pose lifting.

  Args:
      keypoints (ndarray[K, 2 or 3]): 2D keypoints to be transformed.
      pose_det_dataset, (str): Name of the dataset for 2D pose detector.
      pose_lift_dataset (str): Name of the dataset for pose lifter model.

  Returns:
      ndarray[K, 2 or 3]: the transformed 2D keypoints.
  """
  assert pose_lift_dataset in ["Body3DH36MDataset", "Body3DMpiInf3dhpDataset"], (
    "`pose_lift_dataset` should be `Body3DH36MDataset` "
    f"or `Body3DMpiInf3dhpDataset`, but got {pose_lift_dataset}."
  )

  coco_style_datasets = [
    "TopDownCocoDataset",
    "TopDownPoseTrack18Dataset",
    "TopDownPoseTrack18VideoDataset",
  ]
  keypoints_new = np.zeros((17, keypoints.shape[1]), dtype=keypoints.dtype)

  # print(f"{pose_lift_dataset} {pose_det_dataset}")
  if pose_lift_dataset == "Body3DH36MDataset":
    if pose_det_dataset in ["TopDownH36MDataset"]:
      keypoints_new = keypoints
    elif pose_det_dataset in coco_style_datasets:
      # pelvis (root) is in the middle of l_hip and r_hip
      keypoints_new[0] = (keypoints[11] + keypoints[12]) / 2
      # thorax is in the middle of l_shoulder and r_shoulder
      keypoints_new[8] = (keypoints[5] + keypoints[6]) / 2
      # spine is in the middle of thorax and pelvis
      keypoints_new[7] = (keypoints_new[0] + keypoints_new[8]) / 2
      # in COCO, head is in the middle of l_eye and r_eye
      # in PoseTrack18, head is in the middle of head_bottom and head_top
      keypoints_new[10] = (keypoints[1] + keypoints[2]) / 2
      # rearrange other keypoints
      keypoints_new[[1, 2, 3, 4, 5, 6, 9, 11, 12, 13, 14, 15, 16]] = keypoints[
        [12, 14, 16, 11, 13, 15, 0, 5, 7, 9, 6, 8, 10]
      ]
    elif pose_det_dataset in ["TopDownAicDataset"]:
      # pelvis (root) is in the middle of l_hip and r_hip
      keypoints_new[0] = (keypoints[9] + keypoints[6]) / 2
      # thorax is in the middle of l_shoulder and r_shoulder
      keypoints_new[8] = (keypoints[3] + keypoints[0]) / 2
      # spine is in the middle of thorax and pelvis
      keypoints_new[7] = (keypoints_new[0] + keypoints_new[8]) / 2
      # neck base (top end of neck) is 1/4 the way from
      # neck (bottom end of neck) to head top
      keypoints_new[9] = (3 * keypoints[13] + keypoints[12]) / 4
      # head (spherical centre of head) is 7/12 the way from
      # neck (bottom end of neck) to head top
      keypoints_new[10] = (5 * keypoints[13] + 7 * keypoints[12]) / 12

      keypoints_new[[1, 2, 3, 4, 5, 6, 11, 12, 13, 14, 15, 16]] = keypoints[
        [6, 7, 8, 9, 10, 11, 3, 4, 5, 0, 1, 2]
      ]
    elif pose_det_dataset in ["TopDownCrowdPoseDataset"]:
      # pelvis (root) is in the middle of l_hip and r_hip
      keypoints_new[0] = (keypoints[6] + keypoints[7]) / 2
      # thorax is in the middle of l_shoulder and r_shoulder
      keypoints_new[8] = (keypoints[0] + keypoints[1]) / 2
      # spine is in the middle of thorax and pelvis
      keypoints_new[7] = (keypoints_new[0] + keypoints_new[8]) / 2
      # neck base (top end of neck) is 1/4 the way from
      # neck (bottom end of neck) to head top
      keypoints_new[9] = (3 * keypoints[13] + keypoints[12]) / 4
      # head (spherical centre of head) is 7/12 the way from
      # neck (bottom end of neck) to head top
      keypoints_new[10] = (5 * keypoints[13] + 7 * keypoints[12]) / 12

      keypoints_new[[1, 2, 3, 4, 5, 6, 11, 12, 13, 14, 15, 16]] = keypoints[
        [7, 9, 11, 6, 8, 10, 0, 2, 4, 1, 3, 5]
      ]
    else:
      msg = f"unsupported conversion between {pose_lift_dataset} and {pose_det_dataset}"
      raise NotImplementedError(msg)

  elif pose_lift_dataset == "Body3DMpiInf3dhpDataset":
    if pose_det_dataset in coco_style_datasets:
      # pelvis (root) is in the middle of l_hip and r_hip
      keypoints_new[14] = (keypoints[11] + keypoints[12]) / 2
      # neck (bottom end of neck) is in the middle of
      # l_shoulder and r_shoulder
      keypoints_new[1] = (keypoints[5] + keypoints[6]) / 2
      # spine (centre of torso) is in the middle of neck and root
      keypoints_new[15] = (keypoints_new[1] + keypoints_new[14]) / 2

      # in COCO, head is in the middle of l_eye and r_eye
      # in PoseTrack18, head is in the middle of head_bottom and head_top
      keypoints_new[16] = (keypoints[1] + keypoints[2]) / 2

      if "PoseTrack18" in pose_det_dataset:
        keypoints_new[0] = keypoints[1]
        # don't extrapolate the head top confidence score
        keypoints_new[16, 2] = keypoints_new[0, 2]
      else:
        # head top is extrapolated from neck and head
        keypoints_new[0] = (4 * keypoints_new[16] - keypoints_new[1]) / 3
        # don't extrapolate the head top confidence score
        keypoints_new[0, 2] = keypoints_new[16, 2]
      # arms and legs
      keypoints_new[2:14] = keypoints[[6, 8, 10, 5, 7, 9, 12, 14, 16, 11, 13, 15]]
    elif pose_det_dataset in ["TopDownAicDataset"]:
      # head top is head top
      keypoints_new[0] = keypoints[12]
      # neck (bottom end of neck) is neck
      keypoints_new[1] = keypoints[13]
      # pelvis (root) is in the middle of l_hip and r_hip
      keypoints_new[14] = (keypoints[9] + keypoints[6]) / 2
      # spine (centre of torso) is in the middle of neck and root
      keypoints_new[15] = (keypoints_new[1] + keypoints_new[14]) / 2
      # head (spherical centre of head) is 7/12 the way from
      # neck (bottom end of neck) to head top
      keypoints_new[16] = (5 * keypoints[13] + 7 * keypoints[12]) / 12
      # arms and legs
      keypoints_new[2:14] = keypoints[0:12]
    elif pose_det_dataset in ["TopDownCrowdPoseDataset"]:
      # head top is top_head
      keypoints_new[0] = keypoints[12]
      # neck (bottom end of neck) is in the middle of
      # l_shoulder and r_shoulder
      keypoints_new[1] = (keypoints[0] + keypoints[1]) / 2
      # pelvis (root) is in the middle of l_hip and r_hip
      keypoints_new[14] = (keypoints[7] + keypoints[6]) / 2
      # spine (centre of torso) is in the middle of neck and root
      keypoints_new[15] = (keypoints_new[1] + keypoints_new[14]) / 2
      # head (spherical centre of head) is 7/12 the way from
      # neck (bottom end of neck) to head top
      keypoints_new[16] = (5 * keypoints[13] + 7 * keypoints[12]) / 12
      # arms and legs
      keypoints_new[2:14] = keypoints[[1, 3, 5, 0, 2, 4, 7, 9, 11, 6, 8, 10]]

    else:
      raise NotImplementedError(
        f"unsupported conversion between {pose_lift_dataset} and " f"{pose_det_dataset}"
      )

  return keypoints_new


# @numba.jit()
def predict(
  det_config: str = f"{MMPOSE_FOLDER}demo/mmdetection_cfg/faster_rcnn_r50_fpn_coco.py",
  det_checkpoint: str = f"{MODELS_FOLDER}faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth",
  pose_detector_config: Optional[
    str
  ] = f"{MMPOSE_FOLDER}configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/hrnet_w48_coco_256x192.py",
  pose_detector_checkpoint: Optional[
    str
  ] = f"{MODELS_FOLDER}hrnet_w48_coco_256x192-b9e0b3ab_20200708.pth",
  pose_lifter_config: str = f"{MMPOSE_FOLDER}configs/body/3d_kpt_sview_rgb_vid/video_pose_lift/h36m/videopose3d_h36m_243frames_fullconv_supervised_cpn_ft.py",
  pose_lifter_checkpoint: str = f"{MODELS_FOLDER}videopose_h36m_243frames_fullconv_supervised_cpn_ft-88f5abbb_20210527.pth",
  video_path: str = f"{INTERIM_FOLDER}short.mp4",
  subclip: bool = False,
  subclip_start: int = 0,
  subclip_end: int = 1,
  occlude: bool = False,
  occlusion_proportion_h: Optional[float] = 0.35,
  occlusion_proportion_w: Optional[float] = 1.0,
  occlusion_top_left: Tuple[int, int] = (0, 0),
  rebase_keypoint_height=True,
  norm_pose_2d=True,
  num_instances: int = -1,
  show=False,
  out_video_root: str = PROCESSED_FOLDER,
  device="cuda:0",
  det_cat_id: int = 1,
  bbox_thr: float = 0.9,
  # kpt_thr: float = 0.3,
  use_oks_tracking=True,
  tracking_thr: float = 0.3,
  radius: int = 8,
  thickness: int = 2,
  smooth=True,
  smooth_filter_cfg: str = f"{MMPOSE_FOLDER}configs/_base_/filters/one_euro.py",
  use_multi_frames=False,
  online=False,
):

  assert has_mmdet, "Please install mmdet to run the demo."
  assert show or (out_video_root != "")
  assert det_config is not None
  assert det_checkpoint is not None

  video_name = video_path.split("/")[-1].split(".")[0]  # without "mp4"
  output_path = f"{out_video_root}{video_name}.mp4"

  if subclip:
    print("Subclipping video...")

    output_path = output_path.replace(".mp4", "_subclip.mp4")
    subclip_video(
      video_path=video_path,
      output_path=output_path,
      start=subclip_start,
      end=subclip_end,
    )

    video_path = output_path

  if occlude:
    print("Occluding video region...")

    output_path = output_path.replace(".mp4", "_occl.mp4")
    occlude_video_region(
      video_path=video_path,
      output_path=output_path,
      proportion_h=occlusion_proportion_h,
      proportion_w=occlusion_proportion_w,
      top_left=occlusion_top_left,
    )
    video_path = output_path

  video = mmcv.VideoReader(video_path)
  assert video.opened, f"Failed to load video file {video_path}"

  # First stage: 2D pose detection
  print("Stage 1: 2D pose detection.")

  if debug:
    print(det_config)
    print(det_checkpoint)
    print(device.lower())

  print("Initializing model...")
  person_det_model = init_detector(det_config, det_checkpoint, device=device.lower())

  pose_det_model: TopDown = init_pose_model(
    pose_detector_config, pose_detector_checkpoint, device=device.lower()
  )

  assert isinstance(pose_det_model, TopDown), (
    'Only "TopDown"' "model is supported for the 1st stage (2D pose detection)"
  )

  # frame index offsets for inference, used in multi-frame inference setting
  if use_multi_frames:
    assert "frame_indices_test" in pose_det_model.cfg.data.test.data_cfg
    indices = pose_det_model.cfg.data.test.data_cfg["frame_indices_test"]

  pose_det_dataset = pose_det_model.cfg.data["test"]["type"]
  # get datasetinfo
  dataset_info = pose_det_model.cfg.data["test"].get("dataset_info", None)
  if dataset_info is None:
    warnings.warn(
      "Please set `dataset_info` in the config."
      "Check https://github.com/open-mmlab/mmpose/pull/663 for details.",
      DeprecationWarning,
    )
  else:
    dataset_info = DatasetInfo(dataset_info)

  pose_det_results_list = []
  next_id = 0
  pose_det_results = []

  # whether to return heatmap, optional
  return_heatmap = False

  # return the output of some desired layers,
  # e.g. use ('backbone', ) to return backbone feature
  output_layer_names = None

  print("Running 2D pose detection inference...")
  for frame_id, cur_frame in enumerate(mmcv.track_iter_progress(video)):
    pose_det_results_last = pose_det_results

    # test a single image, the resulting box is (x1, y1, x2, y2)
    mmdet_results = inference_detector(person_det_model, cur_frame)

    # keep the person class bounding boxes.
    person_det_results = process_mmdet_results(mmdet_results, det_cat_id)

    if use_multi_frames:
      frames = collect_multi_frames(video, frame_id, indices, online)

    # make person results for current image
    pose_det_results, _ = inference_top_down_pose_model(
      pose_det_model,
      frames if use_multi_frames else cur_frame,
      person_det_results,
      bbox_thr=bbox_thr,
      format="xyxy",
      dataset=pose_det_dataset,
      dataset_info=dataset_info,
      return_heatmap=return_heatmap,
      outputs=output_layer_names,
    )

    # get track id for each person instance
    pose_det_results, next_id = get_track_id(
      pose_det_results,
      pose_det_results_last,
      next_id,
      use_oks=use_oks_tracking,
      tracking_thr=tracking_thr,
    )

    pose_det_results_list.append(copy.deepcopy(pose_det_results))

  # Second stage: Pose lifting
  print("Stage 2: 2D-to-3D pose lifting.")

  print("Initializing model...")
  pose_lift_model = init_pose_model(
    config=pose_lifter_config,
    checkpoint=pose_lifter_checkpoint,
    device=device.lower(),
  )

  msg = 'Only "PoseLifter" model is supported for the 2nd stage ' "(2D-to-3D lifting)"
  assert isinstance(pose_lift_model, PoseLifter), msg

  pose_lift_dataset = pose_lift_model.cfg.data["test"]["type"]

  if out_video_root == "":
    save_out_video = False
  else:
    os.makedirs(out_video_root, exist_ok=True)
    save_out_video = True

  if save_out_video:
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    fps = video.fps
    writer = None

  # convert keypoint definition
  for pose_det_results in pose_det_results_list:
    for res in pose_det_results:
      keypoints = res["keypoints"]
      res["keypoints"] = convert_keypoint_definition(
        keypoints=keypoints,
        pose_det_dataset=pose_det_dataset,
        pose_lift_dataset=pose_lift_dataset,
      )

  # load temporal padding config from model.data_cfg
  if hasattr(pose_lift_model.cfg, "test_data_cfg"):
    data_cfg = pose_lift_model.cfg.test_data_cfg
  else:
    data_cfg = pose_lift_model.cfg.data_cfg

  # build pose smoother for temporal refinement
  if smooth:
    smoother = Smoother(
      filter_cfg=smooth_filter_cfg, keypoint_key="keypoints", keypoint_dim=2
    )
  else:
    smoother = None

  num_instances = num_instances
  pose_lift_dataset_info = pose_lift_model.cfg.data["test"].get("dataset_info", None)
  if pose_lift_dataset_info is None:
    warnings.warn(
      "Please set `dataset_info` in the config."
      "Check https://github.com/open-mmlab/mmpose/pull/663 for details.",
      DeprecationWarning,
    )
  else:
    pose_lift_dataset_info = DatasetInfo(pose_lift_dataset_info)

  global pos_det_results
  pos_det_results = pose_det_results_list

  print("Running 2D-to-3D pose lifting inference...")
  for i, pose_det_results in enumerate(mmcv.track_iter_progress(pose_det_results_list)):
    # extract and pad input pose2d sequence
    pose_results_2d = extract_pose_sequence(
      pose_det_results_list,
      frame_idx=i,
      causal=data_cfg.causal,
      seq_len=data_cfg.seq_len,
      step=data_cfg.seq_frame_interval,
    )

    # smooth 2d results
    if smoother:
      print("Smoothing 2D pose results...")
      pose_results_2d = smoother.smooth(pose_results_2d)

    # 2D-to-3D pose lifting
    pose_lift_results = inference_pose_lifter_model(
      model=pose_lift_model,
      pose_results_2d=pose_results_2d,
      dataset=pose_lift_dataset,
      dataset_info=pose_lift_dataset_info,
      with_track_id=True,
      image_size=video.resolution,
      norm_pose_2d=norm_pose_2d,
    )

    # Create current date for filename
    dt_now = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Pose processing
    pose_lift_results_vis = []
    for idx, res in enumerate(pose_lift_results):
      keypoints_3d = res["keypoints_3d"]

      # exchange y,z-axis, and then reverse the direction of x,z-axis
      keypoints_3d = keypoints_3d[..., [0, 2, 1]]
      keypoints_3d[..., 0] = -keypoints_3d[..., 0]
      keypoints_3d[..., 2] = -keypoints_3d[..., 2]

      # rebase height (z-axis)
      if rebase_keypoint_height:
        keypoints_3d[..., 2] -= np.min(keypoints_3d[..., 2], axis=-1, keepdims=True)
      res["keypoints_3d"] = keypoints_3d

      # add title
      det_res = pose_det_results[idx]
      instance_id = det_res["track_id"]
      res["title"] = f"Prediction ({instance_id})"

      # only visualize the target frame
      res["keypoints"] = det_res["keypoints"]
      res["bbox"] = det_res["bbox"]
      res["track_id"] = instance_id
      pose_lift_results_vis.append(res)

    global results
    results = pose_lift_results_vis

    global info
    info = pose_lift_dataset_info

    global dataset
    dataset = pose_lift_dataset

    # Visualization
    if num_instances < 0:
      num_instances = len(pose_lift_results_vis)

    img_vis = vis_3d_pose_result(
      model=pose_lift_model,
      result=pose_lift_results_vis,
      img=video[i],
      dataset=pose_lift_dataset,
      dataset_info=pose_lift_dataset_info,
      out_file=None,
      radius=radius,
      thickness=thickness,
      num_instances=num_instances,
      show=show,
    )

    if save_out_video:
      if writer is None:
        writer = cv2.VideoWriter(
          filename=osp.join(out_video_root, f"vis_{dt_now}_{osp.basename(video_path)}"),
          fourcc=fourcc,
          fps=fps,
          frameSize=(img_vis.shape[1], img_vis.shape[0]),
        )
      writer.write(img_vis)

  if save_out_video:
    writer.release()
