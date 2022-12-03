import cv2
import time
import torch
import argparse
import numpy as np
from utils.datasets import letterbox
from utils.torch_utils import select_device
from models.experimental import attempt_load
from utils.plots import output_to_keypoint, plot_skeleton_kpts
from utils.general import non_max_suppression_kpt, strip_optimizer
from torchvision import transforms
import json
import csv

from trainer import getAngle, drawAngle, getCoordinates, getImportantAngles, drawImportantAngles, getXY, drawImportantAngleText, getImportantDistances, getImportantCoordinates
from PIL import ImageFont, ImageDraw, Image


@torch.no_grad()
def run(sourcePath, outputPath, poseweights='yolov7-w6-pose.pt', device='cpu'):

    path = sourcePath
    ext = path.split('/')[-1].split('.')[-1].strip().lower()
    if ext in ["mp4", "webm", "avi"] or ext not in ["mp4", "webm", "avi"] and ext.isnumeric():
        input_path = int(path) if path.isnumeric() else path
        device = select_device(opt.device)
        half = device.type != 'cpu'
        model = attempt_load(poseweights, map_location=device)
        _ = model.eval()

        cap = cv2.VideoCapture(input_path)

        if (cap.isOpened() == False):
            print('Error while trying to read video. Please check path again')

        frame_width, frame_height = int(cap.get(3)), int(cap.get(4))
        vid_write_image = letterbox(cap.read()[1], (frame_width), stride=64, auto=True)[0]
        resize_height, resize_width = vid_write_image.shape[:2]
        out_video_name = "output" if path.isnumeric else f"{input_path.split('/')[-1].split('.')[0]}"
        out = cv2.VideoWriter(f"{out_video_name}_result4.mp4", cv2.VideoWriter_fourcc(*'mp4v'), 30, (resize_width, resize_height))

        frame_count, total_fps = 0, 0
        # Used to count pushups and keep track of direction
        bcount = 0
        direction = 0
        # Used to store points, angles, distances
        output_data = []
        while cap.isOpened:
            # NOTE: THIS IS TO BREAK OUT AFTER ONLY 5 FRAMES
            # SAVES TIME!!!!
            if (frame_count > 1):
                break

            print(f"Frame {frame_count} Processing")
            ret, frame = cap.read()
            if ret:
                orig_image = frame

                # preprocess image
                image = cv2.cvtColor(orig_image, cv2.COLOR_BGR2RGB)
                image = letterbox(image, (frame_width), stride=64, auto=True)[0]
                image_ = image.copy()
                image = transforms.ToTensor()(image)
                image = torch.tensor(np.array([image.numpy()]))

                image = image.to(device)
                image = image.float()
                start_time = time.time()

                # Get model outputs form image
                with torch.no_grad():
                    output, _ = model(image)
                output = non_max_suppression_kpt(output, 0.25, 0.65, nc=model.yaml['nc'], nkpt=model.yaml['nkpt'], kpt_label=True)
                output = output_to_keypoint(output)

                img = image[0].permute(1, 2, 0) * 255
                img = img.cpu().numpy().astype(np.uint8)
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

                # This loops through each person in the output (likely just 1)
                for idx in range(output.shape[0]):

                    # NOTE: Below is used to plot entire skeleton
                    # plot_skeleton_kpts(img, output[idx, 7:].T, 3)

                    # Turn outputs to easier-to-use coordinates
                    kpts = output[idx, 7:].T
                    coordinates = getCoordinates(kpts)
                    
                    # Get important angles/distances from coordinates
                    impAngles = getImportantAngles(coordinates)
                    impDistances = getImportantDistances(coordinates)
                    impCoordinates = getImportantCoordinates(coordinates)
                    print("IMPORTANT ANGLES")
                    print(impAngles)
                    print("IMPORTANT DISTANCES")
                    print(impDistances)
                    print("IMPORTANT COORDINATES")
                    print(impCoordinates)

                    # Store angles, distances, coordinates in output_data
                    concat = impAngles + impDistances + impCoordinates
                    output_data.append(concat)
                    print("CONCATENATED")
                    print(concat)
                    print("OUTPUT DATA")
                    print(output_data)

                    # Draw angles on image
                    drawImportantAngles(img, coordinates)
                    print("Important angles: right arm, left arm, right shoulder, left shoulder")
                    print(impAngles)
                    
                    # For progress bar
                    angle = getAngle(coordinates, (5, 7, 9))
                    percentage = np.interp(angle, (210, 290), (0, 100))
                    bar = np.interp(angle, (220, 290), (int(frame_height) - 100, 100))

                    # checks for number of pushups
                    if percentage == 100:
                        if direction == 0:
                            bcount += 0.5
                            direction = 1
                    if percentage == 0:
                        if direction == 1:
                            bcount += 0.5
                            direction = 0

                    # This is for drawing the progress bar
                    color = (254, 118, 136)
                    cv2.line(img, (100, 100), (100, int(frame_height) - 100),
                              (255, 255, 255), 30)
                    cv2.line(img, (100, int(bar)),
                              (100, int(frame_height) - 100), color, 30)
                    if (int(percentage) < 10):
                        cv2.line(img, (155, int(bar)),
                                  (190, int(bar)), (254, 118, 136), 40)
                    elif (int(percentage) >= 10 and (int(percentage) < 100)):
                        cv2.line(img, (155, int(bar)),
                                  (200, int(bar)), (254, 118, 136), 40)
                    else:
                        cv2.line(img, (155, int(bar)),
                                  (210, int(bar)), (254, 118, 136), 40)

                    # Used for drawing on image
                    im = Image.fromarray(img)
                    draw = ImageDraw.Draw(im)
                    
                    # Draw angles
                    # le_x, le_y = getXY(coordinates, 7)
                    # draw.text((le_x+50, le_y+50), f"{impAngles[1]}", fill=(255, 255, 255))
                    drawImportantAngleText(draw, coordinates)

                    # Drawing progress bar on the video
                    # draw.rounded_rectangle((fw - 300, (fh // 2) - 100, fw - 50, (fh // 2) + 100), fill=color, radius=40)
                    draw.text((145, int(bar) - 17), f"{int(percentage)}%", fill=(255, 255, 255))
                    draw.text((frame_width - 230, (frame_height // 2) - 100), f"{int(bcount)}", fill=(255, 255, 255))
                    img = np.array(im)

                if ext.isnumeric():
                    cv2.imshow("Detection", img)
                    key = cv2.waitKey(1)
                    if key == ord('c'):
                        break

                end_time = time.time()
                fps = 1 / (end_time - start_time)
                total_fps += fps
                frame_count += 1
                out.write(img)

                print("Wrote image")
            else:
                break

        cap.release()
        with open(outputPath, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerows(output_data)
        avg_fps = total_fps / frame_count
        print(f"Average FPS: {avg_fps:.3f}")


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--sourcePath', type=str, help='path to video or 0 for webcam')
    parser.add_argument('--outputPath', type=str, help='output file location/name')
    parser.add_argument('--poseweights', nargs='+', type=str, default='yolov7-w6-pose.pt', help='model path(s)')
    parser.add_argument('--device', type=str, default='cpu', help='cpu/0,1,2,3(gpu)')
    opt = parser.parse_args()
    return opt


def main(opt):
    run(**vars(opt))


if __name__ == "__main__":
    opt = parse_opt()
    strip_optimizer(opt.device, opt.poseweights)
    main(opt)
