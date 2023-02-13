# from gen6d_pollen.utils.luxonis_wrapper import LuxonisCameraWrapper
from sdk_wrappers.luxonis_wrapper import LuxonisWrapper

from gen6d_pollen.utils.aruco_utils import ArucoUtils
from megapose.utils.megapose_wrapper import MegaposeWrapper
import cv2
from cv2 import aruco
from FramesViewer.viewer import Viewer
import time


def draw_bboxes(im, bboxes):
    for entry in bboxes:
        bbox = entry["bbox"]
        left = int(round(bbox[0]))
        top = int(round(bbox[1]))
        width = int(round(bbox[2]))
        height = int(round(bbox[3]))
        im = cv2.rectangle(im, (left, top), (left + width, top + height), (0, 0, 255), thickness=2)

    return im


def create_bb_from_luxonis(bb2d_luxonis):
    size_x = abs(bb2d_luxonis[2] - bb2d_luxonis[0])
    size_y = abs(bb2d_luxonis[3] - bb2d_luxonis[1])
    x_tl = bb2d_luxonis[0]
    y_tl = bb2d_luxonis[1]

    return [x_tl, y_tl, size_x, size_y]


item_name = "mug_plastoc"

# This is an example, use any camera you want
# cam = LuxonisCameraWrapper()
lux = LuxonisWrapper(visualize=False)

mpw = MegaposeWrapper(
    "/home/pollen/dev/POC-Vision-pipeline/sdk_wrappers/calibration_luxonis.pckl",
    "/home/pollen/dev/megapose6d/data/examples/" + item_name,
    "megapose-1.0-RGB-multi-hypothesis",
)

arucoUtils = ArucoUtils(8, 5, 0.053, 0.0253, aruco.DICT_4X4_50)
board_x, board_y = arucoUtils.getBoardSize()
aruco_board_bounds = [
    [0, 0, 0],
    [board_x, 0, 0],
    [board_x, board_y, 0],
    [0, board_y, 0],
]

run_inference = False
fv = Viewer()
fv.start()

fv.createPointsList("aruco_board", aruco_board_bounds, size=10, color=(0, 0, 1))

size = 250
key = -1
vis = None
while True:

    lux.update_detections()
    # Img with the results on it
    detection_frame = lux.detection_frame_rgb
    # Raw frame
    im = lux.frame
    if im is None:
        continue

    # A little bit lame, but we don't save the camera resolution in the calibration file
    #   contrary to the megapose format
    if not mpw.is_camera_resolution_set():
        mpw.set_camera_resolution(im.shape[:2])

    # bbox1 = [im.shape[1] // 2 - size // 2, im.shape[0] // 2 - size // 2, size, size]
    # bbox2 = [im.shape[1] // 2 , im.shape[0] // 2 - size // 2, size, size]

    if "cup" in lux.detections and len(lux.detections["cup"]) != 0:
        bbox = create_bb_from_luxonis(lux.detections["cup"][0]["bb"])

        bboxes = [{"label": item_name, "bbox": bbox}]  # , {"label": "mug_plastoc", "bbox": bbox2}]

        T_world_camera = arucoUtils.get_camera_pose(
            im, lux.get_camera_info().K, lux.get_camera_info().D
        )
        if T_world_camera is not None:

            fv.pushFrame(T_world_camera, "T_world_camera")

        if key == 13:
            # enter
            run_inference = not run_inference

        if run_inference:
            start_get_poses = time.time()
            poses = mpw.get_poses(im, bboxes, run_full_pipe=False)
            end_get_poses = time.time()
            start_get_visualization = time.time()
            vis = mpw.get_visualization()
            end_get_visualization = time.time()

            if T_world_camera is not None:
                T_world_object = T_world_camera @ poses[0]
                fv.pushFrame(T_world_object, "T_world_object")

            print("==========")
            print("Time get_poses: ", end_get_poses - start_get_poses)
            print("Time get_visualization: ", end_get_visualization - start_get_visualization)
            print(poses)
            print("==========")

        if key == ord("p"):
            size += 10
        if key == ord("m"):
            size -= 10

        im = draw_bboxes(im, bboxes)
    cv2.imshow("im", im)
    cv2.imshow("detection_frame", detection_frame)
    if vis is not None:
        cv2.imshow("vis", vis)

    key = cv2.waitKey(1)
