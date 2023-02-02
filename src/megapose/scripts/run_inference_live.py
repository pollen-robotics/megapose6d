# Uses detection stage of gen6d (in order to make a quick live demo of megapose)

import argparse
import os
import cv2
import numpy as np
from pathlib import Path

from gen6d_pollen.utils.realsense_wrapper import RealsenseWrapper
from gen6d_pollen.utils.estimator_wrapper import EstimatorWrapper


from bokeh.io import export_png
from bokeh.plotting import gridplot, show
from bokeh.io.export import get_screenshot_as_png

from megapose.visualization.bokeh_plotter import BokehPlotter
from megapose.visualization.utils import make_contour_overlay
from megapose.datasets.scene_dataset import CameraData, ObjectData
from megapose.inference.utils import make_detections_from_object_data
from megapose.datasets.object_dataset import RigidObject, RigidObjectDataset
from megapose.utils.load_model import NAMED_MODELS, load_named_model
from megapose.lib3d.transform import Transform
from megapose.panda3d_renderer.panda3d_scene_renderer import Panda3dSceneRenderer
from megapose.panda3d_renderer import Panda3dLightData
from megapose.utils.conversion import convert_scene_observation_to_panda3d
from megapose.inference.types import (
    DetectionsType,
    ObservationTensor,
    PoseEstimatesType,
)


parser = argparse.ArgumentParser()
parser.add_argument("--mp_example_dir", type=str, required=True)
parser.add_argument("--mp_model", type=str, default="megapose-1.0-RGB-multi-hypothesis")
parser.add_argument("--g6d_cfg", type=str, default="(gen6d) path to configs dir", required=True)
parser.add_argument("--g6d_database", type=str, help="(gen6d) path to database dir", required=True)
parser.add_argument("--g6d_models", type=str, help="(gen6d) path to models dir", required=True)
args = parser.parse_args()


def draw_bbox(im, bbox):
    left = int(round(bbox[0]))
    top = int(round(bbox[1]))
    width = int(round(bbox[2]))
    height = int(round(bbox[3]))
    im = cv2.rectangle(im, (left, top), (left + width, top + height), (0, 0, 255), thickness=2)

    return im


def get_gen6d_detection_bbox(im, ew: EstimatorWrapper):
    det_im, position, scale = ew.detection(im)
    size = det_im.shape[0]
    bbox = np.concatenate([position - size / 2 * scale, np.full(2, size) * scale])
    bbox[1] += bbox[3] / 4
    bbox[3] /= 2
    return bbox


def load_observation_tensor(camera_data, im_rgb, im_depth=None):
    if im_depth is not None:
        observation = ObservationTensor.from_numpy(
            np.array(im_rgb, dtype=np.uint8), np.array(im_depth, dtype=np.float32), camera_data.K
        )
    else:
        observation = ObservationTensor.from_numpy(
            np.array(im_rgb, dtype=np.uint8), None, camera_data.K
        )
    return observation


def load_detection(bbox, label):
    _bbox = [bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]]

    input_object_data = [{"label": label, "bbox_modal": _bbox}]
    input_object_data = [ObjectData.from_json(d) for d in input_object_data]
    detections = make_detections_from_object_data(input_object_data).cuda()

    return detections


def make_object_dataset(example_dir: Path) -> RigidObjectDataset:
    rigid_objects = []
    mesh_units = "mm"
    object_dirs = (example_dir / "meshes").iterdir()
    for object_dir in object_dirs:
        label = object_dir.name
        mesh_path = None
        for fn in object_dir.glob("*"):
            if fn.suffix in {".obj", ".ply"}:
                assert not mesh_path, f"there multiple meshes in the {label} directory"
                mesh_path = fn
        assert mesh_path, f"couldnt find a obj or ply mesh for {label}"
        rigid_objects.append(RigidObject(label=label, mesh_path=mesh_path, mesh_units=mesh_units))
        # TODO: fix mesh units
    rigid_object_dataset = RigidObjectDataset(rigid_objects)
    return rigid_object_dataset


def make_output_visualization(rgb, camera_data, object_dataset, pose_estimates, detections):

    labels = pose_estimates.infos["label"]
    poses = pose_estimates.poses.cpu().numpy()
    object_datas = [
        ObjectData(label=label, TWO=Transform(pose)) for label, pose in zip(labels, poses)
    ]

    camera_data.TWC = Transform(np.eye(4))

    renderer = Panda3dSceneRenderer(object_dataset)

    camera_data, object_datas = convert_scene_observation_to_panda3d(camera_data, object_datas)
    light_datas = [
        Panda3dLightData(
            light_type="ambient",
            color=((1.0, 1.0, 1.0, 1)),
        ),
    ]
    renderings = renderer.render_scene(
        object_datas,
        [camera_data],
        light_datas,
        render_depth=False,
        render_binary_mask=False,
        render_normals=False,
        copy_arrays=True,
    )[0]

    plotter = BokehPlotter()

    fig_rgb = plotter.plot_image(rgb)
    fig_rgb = plotter.plot_detections(fig_rgb, detections=detections)
    fig_mesh_overlay = plotter.plot_overlay(rgb, renderings.rgb)
    contour_overlay = make_contour_overlay(
        rgb, renderings.rgb, dilate_iterations=1, color=(0, 255, 0)
    )["img"]
    fig_contour_overlay = plotter.plot_image(contour_overlay)
    fig_all = gridplot([[fig_rgb, fig_contour_overlay, fig_mesh_overlay]], toolbar_location=None)
    image = np.array(get_screenshot_as_png(fig_all))
    return image
    # vis_dir = example_dir / "visualizations"
    # vis_dir.mkdir(exist_ok=True)
    # export_png(fig_mesh_overlay, filename=vis_dir / "mesh_overlay.png")
    # export_png(fig_contour_overlay, filename=vis_dir / "contour_overlay.png")
    # export_png(fig_all, filename=vis_dir / "all_results.png")

    # return


cam = RealsenseWrapper(os.path.join(args.g6d_cfg, "calibration.pckl"))
# ew = EstimatorWrapper(args.g6d_cfg, args.g6d_database, args.g6d_models, only_detector=True)

camera_data = CameraData.from_json(
    Path(os.path.join(args.mp_example_dir, "camera_data.json")).read_text()
)
object_dataset = make_object_dataset(Path(args.mp_example_dir))

print("Loading model ...")
pose_estimator = load_named_model(args.mp_model, object_dataset).cuda()
print("Done loading model !")

model_info = NAMED_MODELS[args.mp_model]

label = args.mp_example_dir.strip("/").split("/")[-1]
data_dir = os.getenv("MEGAPOSE_DATA_DIR")
assert data_dir

key = -1
running = True
size = 250
while running:

    im = cam.get_im()

    if im is None:
        continue

    # depth = cam.get_depth()

    # bbox = get_gen6d_detection_bbox(im, ew)

    bbox = [im.shape[1] // 2 - size//2, im.shape[0] // 2 - size//2, size, size]

    # if key == 13:

    observation = load_observation_tensor(camera_data, im).cuda()
    detections = load_detection(bbox, label).cuda()
    print("Running inference")
    output, _ = pose_estimator.run_inference_pipeline(
        observation, detections=detections, **model_info["inference_parameters"]
    )

    viz = make_output_visualization(im, camera_data, object_dataset, output, detections)
    cv2.imshow("viz", viz)

    if key == ord("p"):
        size += 10
    if key == ord("m"):
        size -= 10



    im = draw_bbox(im, bbox)

    cv2.imshow("im", im)

    key = cv2.waitKey(1)

    if key == 27:  # Esc
        running = False
