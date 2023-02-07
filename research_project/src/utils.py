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
