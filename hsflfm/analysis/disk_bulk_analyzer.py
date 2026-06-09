from pathlib import Path
import numpy as np
from hsflfm.util import load_dictionary


# This is a modified and simplified version of bulk_analyzer.py(used with the ant data) intended for use with disk specimens, with helper functions that are called in disk_result_display.ipynb
# Update 6/9/26: animate_vector_field_on_image is a work in progress function, the iamge does not line up yet with the vectors

class DiskBulkAnalyzer:
    def __init__(self, result_folder):
        self.result_folder = Path(result_folder)
        self.result_files = self.find_result_files()
        self.results = []

    def find_result_files(self):
        files = []
        files.extend(self.result_folder.rglob("strike_*_results.json"))
        files.extend(self.result_folder.rglob("strike * results.json"))
        return sorted(files)

    def load_results(self):
        self.results = []
        for filename in self.result_files:
            result = load_dictionary(str(filename))
            self.results.append(result)

        print(f"Loaded {len(self.results)} result files.")

    def summarize_z_displacement(self):
        summaries = []

        for result in self.results:
            specimen = result["specimen_number"]
            strike = result["strike_number"]

            # Shape: points x frames x xyz
            displacements = np.asarray(result["camera_point_displacements"])
            z_disp = displacements[:, :, 2]

            mean_z = np.mean(z_disp, axis=0)
            std_z = np.std(z_disp, axis=0)

            summary = {
                "specimen_number": specimen,
                "strike_number": strike,
                "num_points": z_disp.shape[0],
                "num_frames": z_disp.shape[1],
                "start_mean_z": mean_z[0],
                "end_mean_z": mean_z[-1],
                "total_mean_z_displacement": mean_z[-1] - mean_z[0],
                "max_abs_mean_z_displacement": np.max(np.abs(mean_z - mean_z[0])),
                "mean_z_trace": mean_z,
                "std_z_trace": std_z,
            }

            summaries.append(summary)

        return summaries

    def print_summary(self):
        summaries = self.summarize_z_displacement()

        for s in summaries:
            print("----------------------------------------")
            print(f"Specimen: {s['specimen_number']}")
            print(f"Strike: {s['strike_number']}")
            print(f"Points: {s['num_points']}")
            print(f"Frames: {s['num_frames']}")
            print(f"Total mean z displacement: {s['total_mean_z_displacement']:.4f} mm")
            print(f"Max abs mean z displacement: {s['max_abs_mean_z_displacement']:.4f} mm")

    def animate_trial_3d(
        self,
        specimen_number,
        strike_number=None,
        save_path=None,
        fps=10,
        rotate=True,
    ):
        import numpy as np
        import matplotlib.pyplot as plt
        from matplotlib.animation import FuncAnimation, FFMpegWriter, PillowWriter

        # Find matching result
        matches = [
            r for r in self.results
            if r.get("specimen_number") == specimen_number
            and (strike_number is None or int(r.get("strike_number")) == int(strike_number))
        ]

        if len(matches) == 0:
            print(f"No result found for specimen {specimen_number}")
            return

        result = matches[0]

        start_locations = np.asarray(result["camera_start_locations"], dtype=float)
        displacements = np.asarray(result["camera_point_displacements"], dtype=float)

        # points x frames x xyz
        positions = start_locations[:, None, :] + displacements

        # Convert FLFM coordinates to physical coordinates
        positions[:, :, 2] = -positions[:, :, 2]

        num_points, num_frames, _ = positions.shape

        if save_path is None:
            save_path = self.result_folder / f"{specimen_number}_strike_{int(result['strike_number'])}_3d_animation.mp4"
        else:
            save_path = Path(save_path)

        x_all = positions[:, :, 0]
        y_all = positions[:, :, 1]
        z_all = positions[:, :, 2]

        margin = 0.1 * max(
            np.ptp(x_all),
            np.ptp(y_all),
            np.ptp(z_all),
            1e-6,
        )

        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")

        scat = ax.scatter(
            positions[:, 0, 0],
            positions[:, 0, 1],
            positions[:, 0, 2],
        )

        ax.set_xlim(x_all.min() - margin, x_all.max() + margin)
        ax.set_ylim(y_all.min() - margin, y_all.max() + margin)
        ax.set_zlim(z_all.min() - margin, z_all.max() + margin)

        ax.set_xlabel("X position")
        ax.set_ylabel("Y position")
        ax.set_zlabel("Z position")

        def update(frame):
            current = positions[:, frame, :]

            scat._offsets3d = (
                current[:, 0],
                current[:, 1],
                current[:, 2],
            )

            title = (
                f"{specimen_number} | "
                f"Strike {int(result['strike_number'])} | "
                f"Frame {frame + 1}/{num_frames}"
            )

            ax.set_title(title)

            if rotate:
                ax.view_init(elev=25, azim=45 + frame * 2)

            return scat,

        anim = FuncAnimation(
            fig,
            update,
            frames=num_frames,
            interval=1000 / fps,
            blit=False,
        )

        try:
            writer = FFMpegWriter(fps=fps)
            anim.save(save_path, writer=writer)
        except Exception:
            save_path = save_path.with_suffix(".gif")
            writer = PillowWriter(fps=fps)
            anim.save(save_path, writer=writer)

        plt.close(fig)

        print(f"Saved animation to: {save_path}")
        return save_path


    def animate_z_heatmap(
        self,
        specimen_number,
        strike_number=None,
        save_path=None,
        fps=10,
        invert_z=True,
    ):
        import numpy as np
        import matplotlib.pyplot as plt
        from matplotlib.animation import FuncAnimation, FFMpegWriter, PillowWriter
        from pathlib import Path

        matches = [
            r for r in self.results
            if r.get("specimen_number") == specimen_number
            and (strike_number is None or int(r.get("strike_number")) == int(strike_number))
        ]

        if len(matches) == 0:
            print(f"No result found for specimen {specimen_number}")
            return

        result = matches[0]

        start_locations = np.asarray(result["camera_start_locations"], dtype=float)
        displacements = np.asarray(result["camera_point_displacements"], dtype=float)

        x0 = start_locations[:, 0]
        y0 = start_locations[:, 1]

        z_disp = displacements[:, :, 2]

        if invert_z:
            z_disp = -z_disp

        num_points, num_frames = z_disp.shape

        vmax = np.nanmax(np.abs(z_disp))
        vmin = -vmax

        if save_path is None:
            save_path = self.result_folder / f"{specimen_number}_strike_{int(result['strike_number'])}_z_heatmap.mp4"
        else:
            save_path = Path(save_path)

        fig, ax = plt.subplots()

        scat = ax.scatter(
            x0,
            y0,
            c=z_disp[:, 0],
            s=80,
            vmin=vmin,
            vmax=vmax,
            cmap="coolwarm",
        )

        cbar = fig.colorbar(scat, ax=ax)
        cbar.set_label("Physical Z displacement")

        ax.set_xlabel("X position")
        ax.set_ylabel("Y position")
        ax.set_aspect("equal", adjustable="box")

        def update(frame):
            scat.set_array(z_disp[:, frame])
            ax.set_title(
                f"{specimen_number} | Strike {int(result['strike_number'])} | "
                f"Frame {frame + 1}/{num_frames}"
            )
            return scat,

        anim = FuncAnimation(
            fig,
            update,
            frames=num_frames,
            interval=1000 / fps,
            blit=False,
        )

        try:
            writer = FFMpegWriter(fps=fps)
            anim.save(save_path, writer=writer)
        except Exception:
            save_path = save_path.with_suffix(".gif")
            writer = PillowWriter(fps=fps)
            anim.save(save_path, writer=writer)

        plt.close(fig)

        print(f"Saved heatmap animation to: {save_path}")
        return save_path
    
    def animate_vector_field(
        self,
        specimen_number,
        strike_number=None,
        save_path=None,
        fps=10,
        invert_z=True,
        arrow_scale=1.0,
    ):
        import numpy as np
        import matplotlib.pyplot as plt
        from matplotlib.animation import FuncAnimation, FFMpegWriter, PillowWriter
        from pathlib import Path

        matches = [
            r for r in self.results
            if r.get("specimen_number") == specimen_number
            and (strike_number is None or int(r.get("strike_number")) == int(strike_number))
        ]

        if len(matches) == 0:
            print(f"No result found for specimen {specimen_number}")
            return

        result = matches[0]

        start_locations = np.asarray(result["camera_start_locations"], dtype=float)
        displacements = np.asarray(result["camera_point_displacements"], dtype=float)

        x0 = start_locations[:, 0]
        y0 = start_locations[:, 1]

        dx = displacements[:, :, 0]
        dy = displacements[:, :, 1]
        dz = displacements[:, :, 2]

        if invert_z:
            dz = -dz

        num_points, num_frames = dz.shape

        vmax = np.nanmax(np.abs(dz))
        if vmax == 0:
            vmax = 1

        if save_path is None:
            save_path = self.result_folder / f"{specimen_number}_strike_{int(result['strike_number'])}_vector_field.mp4"
        else:
            save_path = Path(save_path)

        fig, ax = plt.subplots()

        q = ax.quiver(
            x0,
            y0,
            dx[:, 0] * arrow_scale,
            dy[:, 0] * arrow_scale,
            dz[:, 0],
            cmap="coolwarm",
            clim=(-vmax, vmax),
            angles="xy",
            scale_units="xy",
            scale=1,
        )

        cbar = fig.colorbar(q, ax=ax)
        cbar.set_label("Physical Z displacement")

        ax.scatter(x0, y0, s=20)

        pad = 0.2 * max(np.ptp(x0), np.ptp(y0), 1e-6)
        ax.set_xlim(x0.min() - pad, x0.max() + pad)
        ax.set_ylim(y0.min() - pad, y0.max() + pad)

        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("X position")
        ax.set_ylabel("Y position")

        def update(frame):
            q.set_UVC(
                dx[:, frame] * arrow_scale,
                dy[:, frame] * arrow_scale,
                dz[:, frame],
            )

            ax.set_title(
                f"{specimen_number} | Strike {int(result['strike_number'])} | "
                f"Frame {frame + 1}/{num_frames}"
            )

            return q,

        anim = FuncAnimation(
            fig,
            update,
            frames=num_frames,
            interval=1000 / fps,
            blit=False,
        )

        try:
            writer = FFMpegWriter(fps=fps)
            anim.save(save_path, writer=writer)
        except Exception:
            save_path = save_path.with_suffix(".gif")
            writer = PillowWriter(fps=fps)
            anim.save(save_path, writer=writer)

        plt.close(fig)

        print(f"Saved vector field animation to: {save_path}")
        return save_path
    
    def animate_vector_field_on_image(
        self,
        specimen_number,
        strike_number=None,
        camera_key=4,
        metadata_manager=None,
        image_filename=None,
        save_path=None,
        fps=10,
        arrow_scale=10,
        invert_z=True,
        flip_arrow_x=False,
        flip_arrow_y=False,
        swap_arrow_xy=False,
    ):
        import numpy as np
        import matplotlib.pyplot as plt
        from matplotlib.animation import FuncAnimation, FFMpegWriter, PillowWriter
        from pathlib import Path
        from hsflfm.util import load_image_set
        from hsflfm.util import load_dictionary, save_dictionary, MetadataManager

        matches = [
            r for r in self.results
            if r.get("specimen_number") == specimen_number
            and (strike_number is None or int(r.get("strike_number")) == int(strike_number))
        ]

        if len(matches) == 0:
            print(f"No result found for specimen {specimen_number}")
            return

        result = matches[0]

        camera_key = int(camera_key)
        cam_num = camera_key

        if camera_key not in result["match_points"]:
            print(f"Camera key {camera_key} not found.")
            print(f"Available keys: {list(result['match_points'].keys())}")
            return

        if metadata_manager is None:
            print("Please provide metadata_manager.")
            return

        if image_filename is None:
            image_filename = metadata_manager.light_calibration_filename

        alignment_images = load_image_set(
            filename=image_filename,
            calibration_filename=metadata_manager.calibration_filename,
            image_numbers=[cam_num],
        )

        image = alignment_images[cam_num]

        match_points = np.asarray(result["match_points"][camera_key], dtype=float)


        # match_points columns:
        # [x_volume, y_volume, z_volume, x_image, y_image]
        x0 = match_points[:, 3]
        y0 = match_points[:, 4]

        displacements = np.asarray(result["camera_point_displacements"], dtype=float)

        fig, ax = plt.subplots(figsize=(6,6))

        ax.imshow(image, cmap="gray")

        ax.scatter(x0, y0, c="red", s=20)

        plt.show()

        dx = displacements[:, :, 0]
        dy = displacements[:, :, 1]
        dz = displacements[:, :, 2]

        if invert_z:
            dz = -dz

        arrow_dx = dx.copy()
        arrow_dy = dy.copy()

        if swap_arrow_xy:
            arrow_dx, arrow_dy = arrow_dy, arrow_dx

        if flip_arrow_x:
            arrow_dx = -arrow_dx

        if flip_arrow_y:
            arrow_dy = -arrow_dy

        num_points, num_frames = dz.shape

        vmax = np.nanmax(np.abs(dz))
        if vmax == 0:
            vmax = 1

        if save_path is None:
            save_path = (
                self.result_folder
                / f"{specimen_number}_strike_{int(result['strike_number'])}_camera_{camera_key}_vector_overlay.mp4"
            )
        else:
            save_path = Path(save_path)

        fig, ax = plt.subplots(figsize=(6, 6))

        ax.imshow(image, cmap="gray")

        q = ax.quiver(
            x0,
            y0,
            arrow_dx[:, 0] * arrow_scale,
            arrow_dy[:, 0] * arrow_scale,
            dz[:, 0],
            cmap="coolwarm",
            clim=(-vmax, vmax),
            angles="xy",
            scale_units="xy",
            scale=1,
            width=0.006,
        )

        ax.scatter(x0, y0, s=12, c="yellow")

        cbar = fig.colorbar(q, ax=ax)
        cbar.set_label("Physical Z displacement")

        ax.set_title(
            f"{specimen_number} | Strike {int(result['strike_number'])} | "
            f"Camera {camera_key} | Frame 1/{num_frames}"
        )

        ax.set_axis_off()

        def update(frame):
            q.set_UVC(
                arrow_dx[:, frame] * arrow_scale,
                arrow_dy[:, frame] * arrow_scale,
                dz[:, frame],
            )

            ax.set_title(
                f"{specimen_number} | Strike {int(result['strike_number'])} | "
                f"Camera {camera_key} | Frame {frame + 1}/{num_frames}"
            )

            return q,

        anim = FuncAnimation(
            fig,
            update,
            frames=num_frames,
            interval=1000 / fps,
            blit=False,
        )

        try:
            writer = FFMpegWriter(fps=fps)
            anim.save(save_path, writer=writer)
        except Exception:
            save_path = save_path.with_suffix(".gif")
            writer = PillowWriter(fps=fps)
            anim.save(save_path, writer=writer)

        plt.close(fig)

        print(f"Saved vector overlay animation to: {save_path}")
        return save_path
