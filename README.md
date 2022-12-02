# pose-estimation

conda activate yolov7pose

>>> python pushupcounter.py --source "./input/pushup.mp4" --device 0 --curltracker=True 

To draw the skeleton:
>>> python pose-estimate.py --source "./input/pushup.mp4" --device 0 --curltracker=True --drawskeleton=True