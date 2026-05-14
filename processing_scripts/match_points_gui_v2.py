# This GUI is used to manually match points between the three multi-perspective images
# 2024/11/12

# import
from hsflfm.util import MetadataManager, load_dictionary, save_dictionary
from hsflfm.calibration import FLF_System, generate_ss_volume

import os
import cv2
import sys
import torch
import numpy as np

import qtpy.QtWidgets as QtWidgets
import qtpy.QtGui as QtGui
from qtpy.QtCore import Qt

# specify specimen name
# specimen_number = "20260409_B"
specimen_number = "20260511_B"
data_manager = MetadataManager(specimen_number=specimen_number)

# specify if we're selecting alignment points or paint dots
point_type = "paint"
type_list = ["alignment", "paint"]

if point_type not in type_list:
    raise ValueError(f"point type must be one of {type_list}, not {point_type}")

# specify where this will be saved
save_folder = data_manager.alignment_folder
assert os.path.exists(save_folder)

if point_type == "alignment":
    name = "alignment_points"
    point_types = [
        "head_base",
        "eye_tipe",
        "under_eye_ridge",
        "ridge_top",
        "eye_back_tip",
    ]
elif point_type == "paint":
    name = "match_points"
    point_types = None

save_name = save_folder + f"/{name}"

# Load calibration
calibration_filename = data_manager.calibration_filename
assert os.path.exists(calibration_filename)

system = FLF_System(calibration_filename)
info_manager = system.calib_manager
image_shape = info_manager.image_shape

# load images
if point_type == "alignment":
    images = data_manager.light_calibration_images
elif point_type == "paint":
    images = data_manager.get_start_images(strike_number=1)

# heights
heights = torch.linspace(-5, 5, 200, dtype=torch.float32)

# generate volume + grids
volume, grids = generate_ss_volume(
    calibration_filename=data_manager.calibration_filename,
    images=images,
    heights=heights,
)

# normalize
volume = (volume - torch.min(volume)) / (torch.max(volume) - torch.min(volume)) * 255
volume = volume.to(torch.uint8)
volume = volume.numpy()


class FrameViewer(QtWidgets.QWidget):
    def __init__(self, save_name, system, volume, heights, grid_volume, point_types):
        super().__init__()
        self.heights = heights
        self.volume = volume
        self.grid_volume = grid_volume
        self.current_frame = 0
        self.point_types = point_types
        self.point_number = 0
        self.save_name = save_name

        self.system = system
        self.info_manager = system.calib_manager

        camera_numbers = self.info_manager.image_numbers
        if os.path.exists(save_name):
            self.match_points = load_dictionary(save_name)
        else:
            self.match_points = {num: [] for num in camera_numbers}

        self.initUI()

    def initUI(self):
        layout = QtWidgets.QVBoxLayout()

        self.instruction_label = QtWidgets.QLabel()
        layout.addWidget(self.instruction_label)
        if self.point_types is not None:
            self.instruction_label.setText(self.point_types[self.point_number])

        self.graphics_view = QtWidgets.QGraphicsView()
        self.scene = QtWidgets.QGraphicsScene()
        self.graphics_view.setScene(self.scene)
        self.graphics_view.mouseDoubleClickEvent = self.on_double_click
        layout.addWidget(self.graphics_view)

        self.slider = QtWidgets.QSlider()
        self.slider.setOrientation(Qt.Horizontal)
        self.slider.setMinimum(0)
        self.slider.setMaximum(self.volume.shape[0] - 1)
        self.slider.setValue(0)
        self.slider.valueChanged.connect(self.on_slider_change)
        layout.addWidget(self.slider)

        self.height_label = QtWidgets.QLabel()
        layout.addWidget(self.height_label)

        self.setLayout(layout)
        self.update_frame()

    # IMPORTANT: Point overlay for disk configuration must use the same mapping 
    # convention as point selection.
    #
    # When a user double-clicks the refocused volume, on_double_click() does NOT
    # analytically recompute camera coordinates from calibration functions.
    # Instead, it looks up the corresponding camera-space coordinate directly from
    # self.grid_volume at the clicked volume pixel.
    #
    # Therefore, when drawing saved points back onto the volume, we invert that
    # same grid lookup: convert the saved camera pixel back to normalized grid
    # coordinates, then find the nearest matching location in self.grid_volume.
    #
    # This keeps click collection and visual overlay consistent. The older
    # image_to_volume_pixel() method used get_shift_slopes()/get_pixel_shifts()
    # directly, which could disagree with generate_ss_volume() and cause saved
    # points to appear offset from where they were clicked.

    def update_frame(self):
        frame_data = self.volume[self.current_frame]

        frame_image = QtGui.QImage(
            frame_data.data,
            frame_data.shape[1],
            frame_data.shape[0],
            frame_data.shape[1] * 3,
            QtGui.QImage.Format_RGB888,
        )

        pixmap = QtGui.QPixmap.fromImage(frame_image)
        self.scene.clear()
        self.scene.addPixmap(pixmap)

        # draw points using grid-consistent inverse mapping
        if self.match_points:
            first_cam_num = self.system.reference_camera

            for point in self.match_points[first_cam_num]:
                x_cam_pix, y_cam_pix, z_mm, _, _ = point

                # pixel → normalized [-1,1]
                x_cam_norm = (x_cam_pix / (image_shape[0] - 1)) * 2 - 1
                y_cam_norm = (y_cam_pix / (image_shape[1] - 1)) * 2 - 1

                shift_map = self.grid_volume[first_cam_num][self.current_frame]

                # Invert the click mapping by nearest-neighbor search.
                # shift_map stores, for each volume pixel, the normalized camera
                # coordinate sampled from the source image. We find the volume
                # pixel whose stored normalized coordinate is closest to the
                # saved camera pixel.
                dist = (
                    (shift_map[..., 0] - y_cam_norm) ** 2
                    + (shift_map[..., 1] - x_cam_norm) ** 2
                )

                idx = np.argmin(dist)
                x_vol_pix, y_vol_pix = np.unravel_index(idx, dist.shape)

                self.scene.addEllipse(
                    y_vol_pix, x_vol_pix, 2.0, 2.0, QtGui.QPen(Qt.red)
                )

        self.height_label.setText(f"height: {self.heights[self.current_frame]} mm")

        self.graphics_view.fitInView(self.scene.sceneRect(), Qt.KeepAspectRatio)

    def on_slider_change(self, value):
        self.current_frame = value
        self.update_frame()

    def on_double_click(self, event):
        pos = self.graphics_view.mapToScene(event.pos())

        x_vol_pix = int(pos.y())
        y_vol_pix = int(pos.x())
        z_mm = self.heights[self.current_frame]

        print(f"Double-clicked at coordinates: ({x_vol_pix}, {y_vol_pix})")

        for cam_num, volume in enumerate(self.grid_volume):
            shift_map = volume[self.current_frame]

            y_cam_norm, x_cam_norm = shift_map[x_vol_pix, y_vol_pix]

            x_cam_pix = (x_cam_norm + 1) * (image_shape[0] - 1) / 2
            y_cam_pix = (y_cam_norm + 1) * (image_shape[1] - 1) / 2

            values = [
                float(i) for i in [x_cam_pix, y_cam_pix, z_mm, x_vol_pix, y_vol_pix]
            ]

            self.match_points[cam_num].append(values)

        save_dictionary(self.match_points, self.save_name)

        self.point_number += 1
        if self.point_types is not None:
            self.instruction_label.setText(self.point_types[self.point_number])

        self.update_frame()


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)

    viewer = FrameViewer(
        save_name=save_name,
        system=system,
        volume=volume,
        heights=heights,
        grid_volume=grids,
        point_types=point_types,
    )

    viewer.show()
    sys.exit(app.exec_())