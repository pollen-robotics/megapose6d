from megapose.datasets.scene_dataset import CameraData, ObjectData
from megapose.datasets.object_dataset import RigidObject, RigidObjectDataset
from megapose.utils.load_model import NAMED_MODELS, load_named_model
from megapose.inference.types import ObservationTensor
from megapose.inference.utils import make_detections_from_object_data
from megapose.lib3d.transform import Transform
from megapose.panda3d_renderer.panda3d_scene_renderer import Panda3dSceneRenderer
from megapose.panda3d_renderer import Panda3dLightData
from megapose.utils.conversion import convert_scene_observation_to_panda3d
from megapose.visualization.utils import make_contour_overlay

from bokeh.io import export_png
from bokeh.plotting import gridplot, show
from bokeh.io.export import get_screenshot_as_png
from megapose.visualization.bokeh_plotter import BokehPlotter

import os
from pathlib import Path
import numpy as np


class MegaposeWrapper:
    def __init__(self, camera_calibration_path: str, object_dir: str, model_name: str):
        self._camera_data = CameraData.from_pickle(camera_calibration_path)
        self._object_dataset = self.make_object_dataset(Path(object_dir))

        self._pose_estimator = load_named_model(model_name, self._object_dataset).cuda()
        self._model_info = NAMED_MODELS[model_name]

        self._label = object_dir.strip("/").split("/")[-1]
        self._data_dir = os.getenv("MEGAPOSE_DATA_DIR")

        self._last_output = None
        self._last_im = None
        self._last_detections = None

        assert self._data_dir

    # resolution is (h, w)
    def set_camera_resolution(self, resolution):
        self._camera_data.set_resolution(resolution)

    def is_camera_resolution_set(self):
        return self._camera_data.get_resolution() is not None

    #  bboxes -> [{label: "apple", bbox: ...}, ...]
    # Format of bbox is [xtop, ytop, width, height]
    def get_poses(self, im: np.ndarray, bboxes: list, run_full_pipe=True):

        self._last_im = im
        observation = self.load_observation_tensor(self._camera_data, im).cuda()
        self._last_detections = self.load_detections(bboxes).cuda()

        coarse_estimates = None if run_full_pipe else self._last_output
        self._last_output, _ = self._pose_estimator.run_inference_pipeline(
            observation,
            detections=self._last_detections,
            **self._model_info["inference_parameters"], coarse_estimates=coarse_estimates
        )

        poses = self._last_output.poses.cpu().numpy()

        return poses

    def get_visualization(self):
        viz = self.make_output_visualization(
            self._last_im,
            self._camera_data,
            self._object_dataset,
            self._last_output,
            self._last_detections,
        )

        return viz

    @staticmethod
    def make_object_dataset(object_dir: Path) -> RigidObjectDataset:
        rigid_objects = []
        mesh_units = "mm"
        object_dirs = (object_dir / "meshes").iterdir()
        for object_dir in object_dirs:
            label = object_dir.name
            mesh_path = None
            for fn in object_dir.glob("*"):
                if fn.suffix in {".obj", ".ply"}:
                    assert not mesh_path, f"there multiple meshes in the {label} directory"
                    mesh_path = fn
            assert mesh_path, f"couldnt find a obj or ply mesh for {label}"
            rigid_objects.append(
                RigidObject(label=label, mesh_path=mesh_path, mesh_units=mesh_units)
            )
            # TODO: fix mesh units
        rigid_object_dataset = RigidObjectDataset(rigid_objects)
        return rigid_object_dataset

    @staticmethod
    def load_observation_tensor(camera_data, im_rgb, im_depth=None):
        if im_depth is not None:
            observation = ObservationTensor.from_numpy(
                np.array(im_rgb, dtype=np.uint8),
                np.array(im_depth, dtype=np.float32),
                camera_data.K,
            )
        else:
            observation = ObservationTensor.from_numpy(
                np.array(im_rgb, dtype=np.uint8), None, camera_data.K
            )
        return observation

    @staticmethod
    def load_detections(bboxes):

        input_object_data = []
        for entry in bboxes:
            bbox = entry["bbox"]
            # Convert from [xtop, ytop, width, height] to [x1, y1, x2, y2]
            _bbox = [bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]]
            label = entry["label"]

            input_object_data.append({"label": label, "bbox_modal": _bbox})

        input_object_data = [ObjectData.from_json(d) for d in input_object_data]
        detections = make_detections_from_object_data(input_object_data).cuda()

        return detections

    @staticmethod
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
        fig_all = gridplot(
            [[fig_rgb, fig_contour_overlay, fig_mesh_overlay]], toolbar_location=None
        )
        image = np.array(get_screenshot_as_png(fig_all))
        return image
