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
        output:
            - visualized_frames: a sequence of track-visualized image frames
'''
def visualize_track(img_frames, track_info):
    visualized_frames = img_frames.copy()

    # generate different colors for tracks
    def random_color():    
        random_colors = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        return random_colors

    def dfs_drawTrace(starting_index, track_index, pos_index, color):
        if track_info[track_index][0][pos_index][0] == -1:
            return

        for frame_index in range(starting_index, track_index+2):
            cv2.line(visualized_frames[frame_index], tuple(np.array(track_info[track_index][0][pos_index]).astype(int)), tuple(np.array(track_info[track_index][1][pos_index]).astype(int)), color, 2)
        
        track_info[track_index][0][pos_index] = [-1, -1]

        if track_index == len(track_info) - 1:
            track_info[track_index][1][pos_index] = [-1, -1]
            return
            
        for next_pos_index in range(len(track_info[track_index+1][0])):
            if track_info[track_index+1][0][next_pos_index][0] != -1 and np.all(track_info[track_index][1][pos_index] == track_info[track_index+1][0][next_pos_index]):
                dfs_drawTrace(starting_index, track_index+1, next_pos_index, color)
                track_info[track_index][1][pos_index] = [-1, -1]
                break
    
    for track_index in range(len(track_info)):
        for layer in range(2):
            for pos_index in range(len(track_info[track_index][layer])):
                if track_info[track_index][layer][pos_index][0] != -1:
                    color = random_color()
                    dfs_drawTrace(track_index, track_index, pos_index, color)
    
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

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(dir+'output_video.avi', fourcc, 3.0, (640,480))
    for output in frames:
        output = cv2.resize(output, (640,480))
        out.write(output)
    out.release()