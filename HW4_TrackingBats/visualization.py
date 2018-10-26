import cv2
import numpy as np
from os import makedirs
from os.path import exists
import random
random.seed(1)

'''
    visualize_track(img_frames, track_info, timestamps)
        input:
            - img_frames: a sequence of image frames inside the video
            - track_info: a list of tracks each contains a sequence of coordinates ordered by timestamp
            - timestamps: corresponds to track_info, element[i] = [starting time, ending time] for track i
        output:
            - visualized_frames: a sequence of track-visualized image frames
'''
def visualize_track(img_frames, track_info, timestamps):
    visualized_frames = img_frames.copy()

    # generate different colors for tracks
    random_colors = [(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))\
                    for _ in range(len(track_info))]
    
    for frame_index in range(len(visualized_frames)):
        for track_index in range(len(track_info)):
            # visualize the trace when the object currently exist in the frame
            if timestamps[track_index][0] < frame_index and timestamps[track_index][1] >= frame_index:
                for index in range(1, frame_index - timestamps[track_index][0] + 1):
                    cv2.line(visualized_frames[frame_index],tuple(track_info[track_index][index - 1]),tuple(track_info[track_index][index]),random_colors[track_index],2)
    
    return visualized_frames

# store the frames in a directory
def output_visualization(dir, frames):
    # write the image to the output file
    if not exists(dir):
        makedirs(dir)

    count = 1
    for output in frames:
        cv2.imwrite(dir+str(count)+'.jpeg',output)
        count += 1