import cv2
import sys
sys.path.append('/home/hbvision/mirsaid/LITE')
sys.path.append('/home/hbvision/mirsaid/LITE/yolo_tracking')
from ultralytics import YOLO
import numpy as np
from pathlib import Path
from boxmot import DeepOCSORT, BoTSORT


tracker = BoTSORT(
    model_weights=Path('osnet_x0_25_msmt17.pt'),  # Path to ReID model
    device='cuda:0',  # Use CPU for inference
    fp16=False,  # Use half precision
    appearance_feature_layer='layer14',  # Layer to extract appearance features from
)

yolo_model = YOLO('yolov8m.pt')

vid = cv2.VideoCapture('/home/hbvision/mirsaid/LITE/demo/VIRAT_S_010204_07_000942_000989.mp4')

while True:
    ret, im = vid.read()
    if not ret:
        break

    results = yolo_model.predict(im, appearance_feature_layer='layer14', verbose=False)

    dets = results[0].boxes.data.cpu().numpy()
    appearance_features = results[0].appearance_features.cpu().numpy()

    tracker.update(dets, im, appearance_features) # --> M X (x, y, x, y, id, conf, cls, ind)
    tracker.plot_results(im, show_trajectories=True)

    # break on pressing q or space
    cv2.imshow('BoxMOT detection', im)     
    key = cv2.waitKey(1) & 0xFF
    if key == ord(' ') or key == ord('q'):
        break

vid.release()
cv2.destroyAllWindows()