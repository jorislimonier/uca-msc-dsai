from typing import Optional, Tuple

import cv2
import numpy as np


def subclip_video(video_path: str, output_path: str, start: int, end: int) -> None:
  """
  Shorten a mp4 video to a given start and end time in seconds.

  Args:
      video_path (str): Path to video file.
      output_path (str): Path to output file.
      start (int): Start time in seconds.
      end (int): End time in seconds.

  Example:
      >>> shorten_video("video.mp4", "short.mp4", 0, 10)
  """
  from moviepy.editor import VideoFileClip
  from moviepy.video.VideoClip import VideoClip

  clip = VideoFileClip(video_path)
  clip: VideoClip = clip.subclip(start, end)
  clip.write_videofile(output_path, logger=None)


def occlude_video_region(
  video_path: str,
  output_path: str,
  proportion_h: Optional[float] = 0.35,
  proportion_w: Optional[float] = 1.0,
  top_left: Tuple[int, int] = (0, 0),
) -> None:
  """
  Occlude a region in a video.

  Args:
      `video_path` (str): Path to video file.
      `output_path` (str): Path to output file.
      `proportion_h` (float, optional): Proportion of height to occlude. Defaults to 0.35.
      `proportion_w` (float, optional): Proportion of width to occlude. Defaults to 1.0.
      `top_left` (Tuple[int, int], optional): Top left corner of region to occlude. Defaults to (0, 0).

  Example:
      >>> occlude_video_region(video_path="video.mp4", output_path="occluded.mp4")
  """

  cap = cv2.VideoCapture(video_path)

  first_round = True
  success: bool
  frame_bgr: np.ndarray
  success, frame_bgr = cap.read()

  while success:
    if first_round:
      frame_h, frame_w, _ = frame_bgr.shape

      # Initialize the bottom right corner of the rectangle
      rect_w = int(proportion_w * frame_h)
      rect_h = int(proportion_h * frame_w)
      pt2 = (rect_w, rect_h)

      # Initialize the video writer
      fourcc = cv2.VideoWriter_fourcc(*"mp4v")
      out = cv2.VideoWriter(output_path, fourcc, 50, (frame_w, frame_h))

      first_round = False

    # Draw the rectangle occulusion
    cv2.rectangle(
      img=frame_bgr,
      pt1=top_left,
      pt2=pt2,
      color=(0, 0, 0),
      thickness=-1,
    )

    out.write(frame_bgr)

    success, frame_bgr = cap.read()

  out.release()
  cap.release()


def convert_kp_coco_h36m(kp: np.ndarray) -> np.ndarray:
  """
  Convert keypoints from COCO to H36M format.
  """
  kp_new = np.zeros((17, kp.shape[1]), dtype=kp.dtype)

  # pelvis (root) is in the middle of l_hip and r_hip
  kp_new[0] = (kp[11] + kp[12]) / 2
  # thorax is in the middle of l_shoulder and r_shoulder
  kp_new[8] = (kp[5] + kp[6]) / 2
  # spine is in the middle of thorax and pelvis
  kp_new[7] = (kp_new[0] + kp_new[8]) / 2
  # in COCO, head is in the middle of l_eye and r_eye
  # in PoseTrack18, head is in the middle of head_bottom and head_top
  kp_new[10] = (kp[1] + kp[2]) / 2
  # rearrange other keypoints
  kp_new[[1, 2, 3, 4, 5, 6, 9, 11, 12, 13, 14, 15, 16]] = kp[
    [12, 14, 16, 11, 13, 15, 0, 5, 7, 9, 6, 8, 10]
  ]

  return kp_new
