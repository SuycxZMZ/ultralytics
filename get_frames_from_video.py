import cv2
import os
from pathlib import Path

def extract_frames_by_index(video_path: str, frame_indices: list[int]):
    """
    Extracts specified frames from a video and saves them as images in a new folder.

    Args:
        video_path (str): Path to the video file.
        frame_indices (list[int]): List of frame indices (0-based) to extract.
    """
    video_path = Path(video_path)
    if not video_path.is_file():
        raise FileNotFoundError(f"❌ Video not found: {video_path}")

    # Get video metadata
    video_name = video_path.stem
    output_dir = video_path.parent / f"{video_name}_extracted_frames"
    output_dir.mkdir(exist_ok=True)

    # Open video
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise IOError(f"❌ Cannot open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_indices = sorted(set(i for i in frame_indices if 0 <= i < total_frames))

    if not frame_indices:
        print("⚠️ No valid frame indices provided.")
        cap.release()
        return

    print(f"🎥 Processing video: {video_path.name}")
    print(f"📦 Extracting frames: {frame_indices}")
    print(f"💾 Saving to folder: {output_dir}")

    current_frame = 0
    target_set = set(frame_indices)
    saved_count = 0

    while cap.isOpened() and frame_indices:
        ret, frame = cap.read()
        if not ret:
            break

        if current_frame in target_set:
            output_file = output_dir / f"{video_name}_{current_frame}.jpg"
            cv2.imwrite(str(output_file), frame)
            print(f"✅ Saved frame {current_frame} -> {output_file.name}")
            saved_count += 1
            target_set.remove(current_frame)

        current_frame += 1
        if current_frame > max(frame_indices):
            break

    cap.release()
    print(f"\n✅ Completed: {saved_count} frames saved to {output_dir}")


# Example usage:
if __name__ == "__main__":
    video_path = "/Users/yuansu/Code/ultralytics/test-videos/C0015.MP4"  # Replace with your video path
    desired_frames = [1, ]  # Replace with your frame numbers

    extract_frames_by_index(video_path, desired_frames)