
import cv2

def visualization(tracked_objects, images):
    for img in range(len(images)):
        cv2.imwrite("./output/" + str(img) + '.jpeg', images[img])

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'H264')
    out = cv2.VideoWriter('./output_video.MP4', fourcc, 3.0, (640,480))
    for output in images:
        output = cv2.resize(images[img], (640,480))
        out.write(output)
    out.release()

    return
