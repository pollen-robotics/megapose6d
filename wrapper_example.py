# Standard Library
import time

# Third Party
import cv2
from cv2 import aruco
from FramesViewer.viewer import Viewer
from gen6d_pollen.utils.aruco_utils import ArucoUtils
from gen6d_pollen.utils.realsense_wrapper import RealsenseWrapper

# MegaPose
from megapose.utils.megapose_wrapper import MegaposeWrapper


def draw_bboxes(im, bboxes):
    for entry in bboxes:
        bbox = entry["bbox"]
        left = int(round(bbox[0]))
        top = int(round(bbox[1]))
        width = int(round(bbox[2]))
        height = int(round(bbox[3]))
        im = cv2.rectangle(im, (left, top), (left + width, top + height), (0, 0, 255), thickness=2)

    return im


item_name = "cylindre_onshape"

# This is an example, use any camera you want
cam = RealsenseWrapper("/tmp/aze/calibration.pckl")
mpw = MegaposeWrapper(
    "/tmp/aze/calibration.pckl",
    "/home/antoine/Pollen/INCIA/megapose6d/data/examples/" + item_name,
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

    im = cam.get_im()
    if im is None:
        continue

    # A little bit lame, but we don't save the camera resolution in the calibration file
    #   contrary to the megapose format
    if not mpw.is_camera_resolution_set():
        mpw.set_camera_resolution(im.shape[:2])

    bbox1 = [im.shape[1] // 2 - size // 2, im.shape[0] // 2 - size // 2, size, size]
    # bbox2 = [im.shape[1] // 2 , im.shape[0] // 2 - size // 2, size, size]
    bboxes = [{"label": item_name, "bbox": bbox1}]  # , {"label": "mug_plastoc", "bbox": bbox2}]

    T_world_camera = arucoUtils.get_camera_pose(
        im, cam.get_camera_matrix(), cam.get_distortion_coefficients()
    )
    if T_world_camera is not None:

        fv.pushFrame(T_world_camera, "T_world_camera")

    if key == 13:
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
    if vis is not None:
        cv2.imshow("vis", vis)

    key = cv2.waitKey(1)
