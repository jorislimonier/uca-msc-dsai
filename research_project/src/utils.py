import cv2

def shorten_video(video_path: str, output_path: str, start: int, end: int) -> None:
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

  clip = VideoFileClip(video_path)
  clip = clip.subclip(start, end)
  clip.write_videofile(output_path)


def black_video_region(video_path: str, output_path: str) -> None:
    """
    Black out a region in a video.
    
    Args:
        video_path (str): Path to video file.
        output_path (str): Path to output file.
    
    Example:
        >>> black_video_region("video.mp4", "black.mp4")
    """
    
    cap = cv2.VideoCapture(video_path)
    success, frame_bgr = cap.read()
    
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"), 30, (1920, 1080))
    