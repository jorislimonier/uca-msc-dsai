from typing import List

import cdflib
import cv2
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from constants import *


def overlay_keypoints(
  kp_path: str = f"{RAW_FOLDER}s1/Poses_D2_Positions_S1/S1/MyPoseFeatures/D2_Positions/Directions 1.54138969.cdf",
  video_path: str = f"{RAW_FOLDER}s1/S1/Videos/Directions 1.54138969.mp4",
  max_frame: int = 300,
  show_nth_frame: int = 100,
  kp_color: np.ndarray = np.array([255, 0, 0]),
  debug: bool = False,
) -> List[go.Figure]:
  """
  Overlay keypoints on video frames.

  Args:
      kp_path (str, optional): Path to cdf file. Defaults to f"{RAW_FOLDER}s1/Poses_D2_Positions_S1/S1/MyPoseFeatures/D2_Positions/Directions 1.54138969.cdf".
      video_path (str, optional): Path to video file. Defaults to f"{RAW_FOLDER}s1/S1/Videos/Directions 1.54138969.mp4".
      max_frame (int, optional): Maximum number of frames to overlay keypoints on. Defaults to 300.
      show_nth_frame (int, optional): Only show every nth frame. Defaults to 100.
      kp_color (np.ndarray, optional): Color of keypoints. Defaults to np.array([255, 0, 0]).
      debug (bool, optional): Print debug info. Defaults to False.

  Returns:
      List[go.Figure]: List of plotly figures.

  Example:
      >>> figs = overlay_keypoints()
      >>> figs[0].show()
  """
  cdf = cdflib.CDF(kp_path)

  pose = cdflib.cdf_to_xarray(kp_path).data_vars["Pose"].values

  cap = cv2.VideoCapture(video_path)
  length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

  if debug:
    print(f"{pose.shape = }")  # dim 0 = number of frames - 1
    print(f"{length = }")
    print(f"{cap.get(cv2.CAP_PROP_FRAME_HEIGHT) = }")
    print(f"{cap.get(cv2.CAP_PROP_FRAME_WIDTH) = }")

  success = True
  count = 0  # max frame number

  figs = []

  while success & (count < max_frame):
    coord = pose[count].reshape(32, -1).astype(int)

    success, frame_bgr = cap.read()
    
    if count % show_nth_frame == 0:
      frame = frame_bgr[:, :, ::-1]

      side = 10
      for w in range(side):
        for h in range(side):
          frame[coord[:, 1] - w, coord[:, 0] - h] = kp_color
          frame[coord[:, 1] + w, coord[:, 0] + h] = kp_color
      fig = px.imshow(frame)
      fig.update_layout(
        title=f"Frame {count}",
        xaxis_title="x",
        yaxis_title="y",
        font=dict(family="Courier New, monospace", size=18, color="#7f7f7f"),
      )
      figs.append(fig)

    count += 1

  return figs
