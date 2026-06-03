from hsflfm.config import home_directory
import cv2
import os
from pathlib import Path

# -------------------------
# Settings
# -------------------------
image_folder = Path(r"C:\Users\abhin\Box\Vaduri_Abhinav Jumping Shells\Collected_Disk_Videos\20260526_B")
output_avi = image_folder / "20260526.avi"

# Ordered frame names based on your displacement sequence
# frame_order = [
#     #"20260511_00_0.tiff",  # z = -5 mm
#     #"20260511_02_0.tiff",  # z = -4 mm
#     #"20260511_04_0.tiff",  # z = -3 mm
#     #"20260511_06_0.tiff",  # z = -2 mm
#     #"20260511_08_0.tiff",  # z = -1 mm
#     "20260511_10_0.tiff",  # z =  0 mm
#     "20260511_12_0.tiff",  # z = +1 mm
#     "20260511_14_0.tiff",  # z = +2 mm
#     "20260511_16_0.tiff",  # z = +3 mm
#     "20260511_18_0.tiff",  # z = +4 mm
#     #"20260511_20_0.tiff",  # z = +5 mm
# ]

frame_order = [
    "20260526_B_0_0_0.tiff",
    "20260526_B_0_1_0.tiff",
    "20260526_B_0_2_0.tiff",
    "20260526_B_0_3_0.tiff",
    "20260526_B_0_4_0.tiff",
    "20260526_B_0_5_0.tiff",

    "20260526_B_1_1_0.tiff",
    "20260526_B_1_2_0.tiff",
    "20260526_B_1_3_0.tiff",
    "20260526_B_1_4_0.tiff",
    "20260526_B_1_5_0.tiff",

    "20260526_B_2_1_0.tiff",
    "20260526_B_2_2_0.tiff",
    "20260526_B_2_3_0.tiff",
    "20260526_B_2_4_0.tiff",
    "20260526_B_2_5_0.tiff",

    "20260526_B_3_1_0.tiff",
    "20260526_B_3_2_0.tiff",
    "20260526_B_3_3_0.tiff",
    "20260526_B_3_4_0.tiff",
    "20260526_B_3_5_0.tiff",

    "20260526_B_4_1_0.tiff",
    "20260526_B_4_2_0.tiff",
    "20260526_B_4_3_0.tiff",
    "20260526_B_4_4_0.tiff",
    "20260526_B_4_5_0.tiff",

    "20260526_B_5_1_0.tiff",
    "20260526_B_5_2_0.tiff",
    "20260526_B_5_3_0.tiff",
    "20260526_B_5_4_0.tiff",
    "20260526_B_5_5_0.tiff",

    "20260526_B_6_1_0.tiff",
    "20260526_B_6_2_0.tiff",
    "20260526_B_6_3_0.tiff",
    "20260526_B_6_4_0.tiff",
    "20260526_B_6_5_0.tiff",

    "20260526_B_7_1_0.tiff",
    "20260526_B_7_2_0.tiff",
    "20260526_B_7_3_0.tiff",
    "20260526_B_7_4_0.tiff",
    "20260526_B_7_5_0.tiff",
]

fps = 10 # arbitrary for testing; change if needed

# -------------------------
# Load first image to get size
# -------------------------
first_path = image_folder / frame_order[0]
first_img = cv2.imread(str(first_path), cv2.IMREAD_UNCHANGED)

if first_img is None:
    raise FileNotFoundError(f"Could not read first image: {first_path}")

# Convert first image to 8-bit grayscale or BGR if needed
def prepare_frame(img):
    if img is None:
        raise ValueError("Image could not be loaded.")

    # If image is 16-bit or float, normalize to 8-bit
    if img.dtype != "uint8":
        img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)
        img = img.astype("uint8")

    # If grayscale, convert to BGR because AVI writers are usually happier with color frames
    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    return img

first_frame = prepare_frame(first_img)
height, width = first_frame.shape[:2]

# -------------------------
# Create AVI writer
# -------------------------
fourcc = cv2.VideoWriter_fourcc(*"MJPG")
writer = cv2.VideoWriter(str(output_avi), fourcc, fps, (width, height))

if not writer.isOpened():
    raise RuntimeError(f"Could not open video writer for: {output_avi}")

# -------------------------
# Write frames
# -------------------------
for filename in frame_order:
    path = image_folder / filename
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)

    if img is None:
        raise FileNotFoundError(f"Could not read image: {path}")

    frame = prepare_frame(img)

    if frame.shape[:2] != (height, width):
        raise ValueError(f"Image size mismatch for {path}")

    writer.write(frame)

writer.release()

print(f"Saved AVI to: {output_avi}")