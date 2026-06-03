from pathlib import Path
import numpy as np
from hsflfm.util import load_dictionary


class DiskBulkAnalyzer:
    def __init__(self, result_folder):
        self.result_folder = Path(result_folder)
        self.result_files = self.find_result_files()
        self.results = []

    def find_result_files(self):
        return sorted(self.result_folder.glob("*/strike_*_results.json"))

    def load_results(self):
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