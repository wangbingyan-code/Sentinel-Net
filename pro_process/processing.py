import cv2
import os
from tqdm import tqdm


def extract_frames(video_path, output_dir, fps=3):
    # Make sure output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Start capturing the video
    cap = cv2.VideoCapture(video_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Calculate the frame index increment based on the desired fps
    frame_index_increment = int(cap.get(cv2.CAP_PROP_FPS) / fps)

    # Use tqdm for progress bar
    for frame_idx in tqdm(range(0, frame_count, frame_index_increment), desc="Extracting frames"):
        # Set the frame index
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)

        # Read the frame
        ret, frame = cap.read()
        if not ret:
            break  # Break the loop if no frame is returned

        # Save each frame to the output directory
        frame_file = os.path.join(output_dir, f"frame_{frame_idx:05d}.jpg")
        cv2.imwrite(frame_file, frame)

    # Release the video capture object
    cap.release()


video_paths = ['../dataset_wby/data.mp4']

for video_path in video_paths:
    video_name = os.path.basename(video_path).split('.')[0]
    output_dir = f'dataset/{video_name}'
    extract_frames(video_path, output_dir, fps=15)