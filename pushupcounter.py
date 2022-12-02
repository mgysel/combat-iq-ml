#YOLOv7 Push-up Detection Tutorial
#By Augmented Startups
#Visit www.augmentedstartups.com
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

# =1.0=Import=Libraries======================
from trainer import findAngle
from PIL import ImageFont, ImageDraw, Image
# ===========================================

@torch.no_grad()
def run(poseweights='yolov7-w6-pose.pt', source='pose.mp4', device='cpu', curltracker=False, drawskeleton=True):
    path = source
    ext = path.split('/')[-1].split('.')[-1].strip().lower()
    if ext in ["mp4", "webm", "avi"] or ext not in ["mp4", "webm", "avi"] and ext.isnumeric():
        input_path = int(path) if path.isnumeric() else path
        device = select_device(opt.device)
        half = device.type != 'cpu'
        model = attempt_load(poseweights, map_location=device)
        _ = model.eval()

        cap = cv2.VideoCapture(input_path)
        webcam = False

        if (cap.isOpened() == False):
            print('Error while trying to read video. Please check path again')

        fw, fh = int(cap.get(3)), int(cap.get(4))
        if ext.isnumeric():
            webcam = True
            fw, fh = 1280, 768
        vid_write_image = letterbox(
            cap.read()[1], (fw), stride=64, auto=True)[0]

        resize_height, resize_width = vid_write_image.shape[:2]
        out_video_name = "output" if path.isnumeric(
        ) else f"{input_path.split('/')[-1].split('.')[0]}"
        out = cv2.VideoWriter(f"{out_video_name}_kpt.mp4", cv2.VideoWriter_fourcc(
            *'mp4v'), 30, (resize_width, resize_height))
        if webcam:
            out = cv2.VideoWriter(f"{out_video_name}_kptsr.mp4", cv2.VideoWriter_fourcc(
                *'mp4v'), 30, (fw, fh))

        frame_count, total_fps = 0, 0
        # =2.1=Variables=========================================
        bcount = 0
        direction = 0

        # =2.2=Load custom font=========================================
        fontpath = "sfpro.ttf"
        font = ImageFont.truetype(fontpath, 32)
        font1 = ImageFont.truetype(fontpath, 160)
        #===========================================
        while cap.isOpened:

            print(f"Frame {frame_count} Processing")
            ret, frame = cap.read()
            if ret:
                orig_image = frame

                # Preprocess Image
                image = cv2.cvtColor(orig_image, cv2.COLOR_BGR2RGB)
                if webcam:
                    image = cv2.resize(image, (fw, fh), interpolation=cv2.INTER_LINEAR)
                image = letterbox(image, (fw), stride=64, auto=True)[0]
                image = transforms.ToTensor()(image)
                image = torch.tensor(np.array([image.numpy()]))

                image = image.to(device)
                image = image.float()
                start_time = time.time()

                # Obtain keypoints
                with torch.no_grad():
                    output, _ = model(image)

                output = non_max_suppression_kpt(
                    output, 0.5, 0.65, nc=model.yaml['nc'], nkpt=model.yaml['nkpt'], kpt_label=True)
                output = output_to_keypoint(output)
                print("OUTPUT")
                print(output)
                img = image[0].permute(1, 2, 0) * 255
                img = img.cpu().numpy().astype(np.uint8)

                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                
                # =3.0=Pushup=Tracking=&=Counting===================
                if curltracker:
                    print("CURLTRACKER")
                    for idx in range(output.shape[0]):
                        # Get keypoints and transpose
                        kpts = output[idx, 7:].T
                        # NOTE: KEYPOINT MAP HERE: https://github.com/JRKagumba/2D-video-pose-estimation-yolov7
                        # Right arm =(5,7,9), left arm = (6,8,10)
                        # Get angles, map from 0 to 100
                        angle = findAngle(img, kpts, 5, 7, 9, draw=True)
                        print("ANGLE")
                        print(angle)
                        # For progress bar
                        percentage = np.interp(angle, (210, 290), (0, 100))
                        bar = np.interp(angle, (220, 290), (int(fh) - 100, 100))

                        color = (254, 118, 136)
                        # check for the Pushup Press
                        if percentage == 100:
                            if direction == 0:
                                bcount += 0.5
                                direction = 1
                        if percentage == 0:
                            if direction == 1:
                                bcount += 0.5
                                direction = 0

                        # This is for drawing the progress bar
                        cv2.line(img, (100, 100), (100, int(fh) - 100),
                                 (255, 255, 255), 30)
                        cv2.line(img, (100, int(bar)),
                                 (100, int(fh) - 100), color, 30)

                        # This is for drawing the progress bar
                        if (int(percentage) < 10):
                            cv2.line(img, (155, int(bar)),
                                     (190, int(bar)), (254, 118, 136), 40)
                        elif (int(percentage) >= 10 and (int(percentage) < 100)):
                            cv2.line(img, (155, int(bar)),
                                     (200, int(bar)), (254, 118, 136), 40)
                        else:
                            cv2.line(img, (155, int(bar)),
                                     (210, int(bar)), (254, 118, 136), 40)

                        # Drawing on the video
                        im = Image.fromarray(img)
                        draw = ImageDraw.Draw(im)
                        # draw.rounded_rectangle((fw - 300, (fh // 2) - 100, fw - 50, (fh // 2) + 100), fill=color, radius=40)
                        draw.text(
                            (145, int(bar) - 17), f"{int(percentage)}%", font=font, fill=(255, 255, 255))

                        draw.text((fw - 230, (fh // 2) - 100),
                                  f"{int(bcount)}", font=font1, fill=(255, 255, 255))
                        img = np.array(im)
                        # ===========================================================

                if drawskeleton:
                    for idx in range(output.shape[0]):
                        plot_skeleton_kpts(img, output[idx, 7:].T, 3)
                # Display Image
                if webcam:
                    cv2.imshow("Detection", img)
                    key = cv2.waitKey(1)
                    if key == ord('c'):
                        break
                else:
                    img_ = img.copy()
                    img_ = cv2.resize(
                        img_, (960, 540), interpolation=cv2.INTER_LINEAR)
                    cv2.imshow("Detection", img_)
                    cv2.waitKey(1)

                end_time = time.time()
                fps = 1 / (end_time - start_time)
                total_fps += fps
                frame_count += 1
                out.write(img)
            else:
                break

        cap.release()
        avg_fps = total_fps / frame_count
        print(f"Average FPS: {avg_fps:.3f}")


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--poseweights', nargs='+', type=str,
                        default='yolov7-w6-pose.pt', help='model path(s)')
    parser.add_argument('--source', type=str,
                        help='path to video or 0 for webcam')
    parser.add_argument('--device', type=str, default='cpu',
                        help='cpu/0,1,2,3(gpu)')
    parser.add_argument('--curltracker', type=bool, default=False,
                        help='set as true to check count bicep curls')
    parser.add_argument('--drawskeleton', type=bool,
                        help='draw all keypoints')

    opt = parser.parse_args()
    return opt


def main(opt):
    run(**vars(opt))


if __name__ == "__main__":
    opt = parse_opt()
    strip_optimizer(opt.device, opt.poseweights)
    main(opt)
