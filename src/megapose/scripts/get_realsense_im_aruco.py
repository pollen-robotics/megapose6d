import pickle
import cv2
from gen6d_pollen.utils.realsense_wrapper import RealsenseWrapper
from gen6d_pollen.utils.aruco_utils import ArucoUtils

rw = RealsenseWrapper("/home/antoine/Pollen/INCIA/gen6d_pollen/data/configs/calibration.pckl")
arucoUtils = ArucoUtils(8, 5, 0.053, 0.0253, cv2.aruco.DICT_4X4_50)
while True:
    im = rw.get_im()
    if im is None:
        continue

    depth = rw.get_depth()

    T_world_camera = arucoUtils.get_camera_pose(
        im, rw.get_camera_matrix(), rw.get_distortion_coefficients()
    )

    cv2.imshow("im", im)
    cv2.imshow("depth", depth)

    key = cv2.waitKey(1)

    if key == 13:
        if T_world_camera is not None:
            print("Saving image image_rgb.png, image_depth.png and camera pose in camera_pose.pckl")
            cv2.imwrite("image_rgb.png", im)
            cv2.imwrite("image_depth.png", depth)
            with open("camera_pose.pckl", "wb") as f:
                pickle.dump(T_world_camera, f)

        else:
            print("No Aruco marker found")
