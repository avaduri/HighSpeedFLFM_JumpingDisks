from matplotlib import pyplot as plt
import numpy as np
import cv2
import math
from hsflfm.util import display_with_points


# take a bunch of lines and condense them to a set of vertical and horizontal lines
def condense_lines(lines, angle_thresh=0.05, r_thresh=20):


    if lines is None:
        return {"horizontal": [], "vertical": []}

    for i in range(lines.shape[0]):
        if abs(lines[i, :, 1] - np.pi) < angle_thresh:
            lines[i, :, 1] = lines[i, :, 1] - np.pi
            lines[i, :, 0] = -lines[i, :, 0]

    angle_dict = {"horizontal": 0, "vertical": np.pi / 2}

    lines_dict = {}

    for direction, angle in angle_dict.items():

        # --- BEFORE angle filtering ---
        angle_mask = abs(lines[:, :, 1] - angle) < angle_thresh

        line_group = np.asarray(
            lines[np.where(abs(lines[:, :, 1] - angle) < angle_thresh)]
        )
        # sort the line group according to r (first index)
        line_group = line_group[line_group[:, 0].argsort()]

        new_line_group = []
        for entry in line_group:
            # make sure we didn't already handle a close value
            if (
                len(new_line_group) > 0
                and len(
                    np.where(
                        abs(np.asarray(new_line_group)[:, 0] - entry[0]) < r_thresh
                    )[0]
                )
                > 0
            ):
                continue

            close_entries = line_group[
                np.where(abs(line_group[:, 0] - entry[0]) < r_thresh)
            ]
            entry = np.mean(close_entries, axis=0)
            new_line_group.append([entry.tolist()])

        lines_dict[direction] = np.asarray(new_line_group).squeeze()

    # this looks like something we should read more later:
    # https://stackoverflow.com/questions/46565975/find-intersection-point-of-two-lines-drawn-using-houghlines-opencv

    return lines_dict


def make_binary_image(image, threshold_values):
    binary = cv2.adaptiveThreshold(
        image,
        255,
        cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY,
        threshold_values["adaptive_threshold_range"],
        2,
    )
    blurred = cv2.medianBlur(binary, threshold_values["blur_range"])

    return blurred

# --- REPLACEMENT get_line_intersect ---

def _to_rho_theta(line):
    """
    Normalize several OpenCV Hough output formats into (rho, theta) floats.
    Accepts:
      - iterable-like [rho, theta] (shape (2,))
      - nested arrays like [[rho, theta]] or (N,1,2)
      - probabilistic line (x1,y1,x2,y2)
    Returns:
      (rho, theta) as floats
    Raises:
      ValueError if the format is not interpretable.
    """
    a = np.asarray(line)

    # scalar (bad)
    if a.ndim == 0:
        raise ValueError("Scalar encountered, cannot interpret as (rho,theta)")

    # If it already flattens to 2 values (rho,theta)
    if a.size == 2:
        r, t = a.ravel()[:2]
        return float(r), float(t)

    # If it's (x1,y1,x2,y2) from HoughLinesP or similar
    if a.size >= 4 and (a.shape[-1] == 4 or a.size == 4):
        flat = a.ravel()[:4]
        x1, y1, x2, y2 = map(float, flat)
        dx = x2 - x1
        dy = y2 - y1
        theta_dir = np.arctan2(dy, dx)           # direction angle of segment
        phi = theta_dir + np.pi / 2.0            # normal angle
        rho = x1 * np.cos(phi) + y1 * np.sin(phi)
        return float(rho), float(phi)

    # If nested and last dim >=2: reshape and take first two
    if a.ndim >= 2:
        lastdim = a.shape[-1]
        if lastdim >= 2:
            flat = a.reshape(-1, lastdim)
            r, t = flat[0, 0], flat[0, 1]
            return float(r), float(t)

    raise ValueError(f"Unrecognized line format: shape={a.shape}, size={a.size}")


def get_line_intersect(line1, line2):
    """
    Compute intersection xy of two lines in a tolerant way.
    Accepts polar (rho,theta) or cartesian (x1,y1,x2,y2) input forms.
    Returns np.array([x, y]) or np.array([np.nan, np.nan]) if lines cannot be parsed or are parallel.
    """
    try:
        rho1, theta1 = _to_rho_theta(line1)
        rho2, theta2 = _to_rho_theta(line2)
    except Exception:
        # could not parse one or both lines -> return NaN intersection (no crash)
        return np.array([np.nan, np.nan])

    # Solve linear system in Hesse normal form
    A = np.array([[np.cos(theta1), np.sin(theta1)],
                  [np.cos(theta2), np.sin(theta2)]], dtype=float)
    b = np.array([rho1, rho2], dtype=float)

    try:
        xy = np.linalg.solve(A, b)
        return xy
    except np.linalg.LinAlgError:
        # parallel or singular
        return np.array([np.nan, np.nan])
# --- END replacement ---


# def get_line_intersect(line1, line2):
#     """Finds the intersection of two lines given in Hesse normal form.

#     Returns closest integer pixel locations.
#     See https://stackoverflow.com/a/383527/5087436
#     """
#     rho1, theta1 = line1  # [0]
#     rho2, theta2 = line2  # [0]
#     A = np.array([[np.cos(theta1), np.sin(theta1)], [np.cos(theta2), np.sin(theta2)]])
#     b = np.array([[rho1], [rho2]])
#     x0, y0 = np.linalg.solve(A, b)
#     #x0, y0 = int(np.round(x0)), int(np.round(y0))
#     #x0 = float(x0)
#     #y0 = float(y0)
#     return [x0, y0]


def find_approx_points(lines_dict):
    all_points = []
    for hline in lines_dict["horizontal"]:
        for vline in lines_dict["vertical"]:
            point = get_line_intersect(hline, vline)
            all_points.append(point)
    return np.asarray(all_points)


# this will return True if the area around center_point is determined to have
# a line along the "axis" direction.
# both of which are in display coordinates
def determine_line_validity(
    image,
    crop_distance,
    center_point,
    esf,
    lsf,
    show=False,
    show_if_failed=False,
):
    is_line = True

    # Step 1: assess standard deviation of surrounding image
    start0 = max(int(center_point[1] - crop_distance), 0)
    end0 = min(int(center_point[1] + crop_distance), image.shape[0])
    start1 = max(int(center_point[0] - crop_distance), 0)
    end1 = min(int(center_point[0] + crop_distance), image.shape[1])

    subimage = image[start0:end0, start1:end1]

    deviation = np.std(subimage)
    deviation_thresh = 1
    if deviation < deviation_thresh:
        is_line = False

    # Step 2: Check whether there is a clear low point in the ESF
    esf = esf - np.mean(subimage)
    esf = esf / np.max(np.abs(subimage))
    m2 = np.min(esf)
    # important note: this was -0.04 for a long time
    m2_thresh = -0.04
    if m2 > m2_thresh:
        is_line = False

    # Step 3: Then check whether LSF takes correct shape right before and after low point in ESF
    # lsf = lsf / np.max(np.abs(subimage))
    # min_avg_number = 5
    # avg_number = 20
    # min_index = np.where(esf == np.min(esf))[0][0]
    # min_index2 = np.where(esf == np.min(esf))[0][-1]
    # # TODO get rid of this
    # if min_index == 0:
    #     min_index = 1

    # if min_index < min_avg_number or min_index2 > len(esf) - min_avg_number:
    #     is_line = False

    # start = max(min_index - avg_number, 0)
    # sum_below = np.sum(lsf[start:min_index])
    # if sum_below > m2 / 3:  # maybe use something else?
    #     is_line = False
    # end = min(min_index2 + 1 + avg_number, len(lsf))
    # sum_above = np.sum(lsf[min_index2:end])
    # if sum_above < -m2 / 3:
    #     is_line = False

    if show or (show_if_failed and not is_line):
        plt.figure()
        plt.plot(esf, label="esf")
        plt.axhline(y=0)
        # plt.axvline(x=min_index, linestyle="--", color="black")
        # plt.axvline(x=min_index2, linestyle="--", color="black")
        # plt.plot(np.arange(avg_number) + min_index + 1, np.ones(avg_number) * sum_above)
        # plt.plot(
        #     np.arange(avg_number) + min_index - avg_number,
        #     np.ones(avg_number) * sum_below,
        # )
        plt.plot(lsf, label="lsf")
        plt.legend()
        plt.show()

        plt.figure()
        plt.imshow(subimage)
        plt.colorbar()
        plt.title(f"is this a line? {is_line}. m2: {m2}. deviation: {deviation}")
        plt.show()

    return is_line


# this is a little more of a brute force approach
def _get_surrounding_indices(line_values, point, ignore_warning=False):
    for i in range(len(line_values) - 1):
        p1 = line_values[i]
        p2 = line_values[i + 1]

        if p1 <= point and p2 >= point:
            return i, i + 1
        if p1 >= point and p2 <= point:
            return i, i + 1

    if not ignore_warning:
        print("may be adjusting center at non-line area")
        print("failing.")
    return -1, -1


def get_surrounding_indices(line_values, point, ignore_warning=False):
    idx0 = np.argmin(np.abs(line_values - point))
    if line_values[idx0] < point:
        if (idx0 + 1) < len(line_values) and line_values[idx0 + 1] > point:
            return idx0, idx0 + 1
        elif line_values[idx0 - 1] > point:
            return idx0 - 1, idx0
        else:
            return _get_surrounding_indices(line_values, point, ignore_warning)
    else:
        if (idx0 + 1) < len(line_values) and line_values[idx0 + 1] < point:
            return idx0, idx0 + 1
        elif line_values[idx0 - 1] < point:
            return idx0 - 1, idx0
        else:
            return _get_surrounding_indices(line_values, point, ignore_warning)


# find the point where a line between two points (left_point and right_point)
# would cross the "intersect" line
def get_line_crossing(left_point, right_point, intersect):
    m1 = (right_point[1] - left_point[1]) / (right_point[0] - left_point[0])
    b1 = left_point[1] - (m1 * left_point[0])

    m2 = 0
    b2 = intersect

    xi = (b1 - b2) / (m2 - m1)
    yi = m1 * xi + b1

    return xi, yi

# def get_lsf_esf(
#     image,
#     center_point,
#     axis=0,
#     avg_num=101,
#     plot_range=75,
#     show=True,
#     return_indices=False,
#     ignore_warning=False,
# ):
#     """
#     Robust version of get_lsf_esf.

#     Returns:
#       - if return_indices True: (lsf, esf, xstart, xend, ystart, yend, subimage)
#       - else: (lsf, esf, subimage)
#     On failure returns the same sentinel as before:
#       - return_indices True -> (-1, -1, -1, -1, -1, -1, subimage)  (or -1s with subimage)
#       - return_indices False -> (-1, -1, subimage)
#     This implementation avoids converting NaN to int and catches numeric failures.
#     """

#     # sentinel-producing helper (same shapes used previously)
#     def _fail(return_indices_local):
#         if return_indices_local:
#             # keep subimage as -1 here to match original fail shape when no subimage
#             return (-1, -1, -1, -1, -1, -1, -1)
#         else:
#             return -1, -1, -1

#     # defensive: minimal checks on input
#     try:
#         if image is None or image.size == 0:
#             return _fail(return_indices)

#         # center_point must be length-2 and finite
#         if center_point is None or len(center_point) < 2:
#             return _fail(return_indices)
#         # convert to floats and verify finite
#         cx = float(center_point[0])
#         cy = float(center_point[1])
#         if not (np.isfinite(cx) and np.isfinite(cy)):
#             return _fail(return_indices)
#     except Exception:
#         return _fail(return_indices)

#     h, w = image.shape[:2]

#     # quick near-edge rejection: require some minimal margin so windowing won't fail
#     # use plot_range/avg_num to estimate required margin
#     try:
#         half_plot = int(max(1, float(plot_range) / 2.0))
#         half_avg = int(max(1, float(avg_num) / 2.0))
#     except Exception:
#         return _fail(return_indices)

#     # Minimum safe margin (pixels); if center lies inside this margin try smaller window later
#     MIN_MARGIN = 2

#     # If center is obviously out-of-bounds, fail early
#     if (cx < 0 - MIN_MARGIN) or (cy < 0 - MIN_MARGIN) or (cx > w + MIN_MARGIN) or (cy > h + MIN_MARGIN):
#         return _fail(return_indices)

#     # Helper to compute clamped integer window indices safely
#     def _clamped_indices(cx_f, cy_f, avg_n, plot_r):
#         # axis=0 vs axis=1 differ in how indices are interpreted in original code
#         if axis == 0:
#             xstart = max(int(round(cy_f - avg_n / 2.0)), 0)
#             xend = min(int(round(cy_f + avg_n / 2.0)), image.shape[0])
#             ystart = max(int(round(cx_f - plot_r / 2.0)), 0)
#             yend = min(int(round(cx_f + plot_r / 2.0)), image.shape[1])
#         else:
#             xstart = max(int(round(cy_f - plot_r / 2.0)), 0)
#             xend = min(int(round(cy_f + plot_r / 2.0)), image.shape[0])
#             ystart = max(int(round(cx_f - avg_n / 2.0)), 0)
#             yend = min(int(round(cx_f + avg_n / 2.0)), image.shape[1])
#         return xstart, xend, ystart, yend

#     # compute primary window
#     xstart, xend, ystart, yend = _clamped_indices(cx, cy, avg_num, plot_range)

#     # If the window is empty (no pixels), optionally try a smaller window; otherwise fail
#     if (xend <= xstart) or (yend <= ystart):
#         # try a smaller window once to handle near-edge but still usable points
#         try:
#             small_plot = max(9, int(plot_range * 0.6))
#             small_avg = max(9, int(avg_num * 0.6))
#             xstart, xend, ystart, yend = _clamped_indices(cx, cy, small_avg, small_plot)
#             if (xend <= xstart) or (yend <= ystart):
#                 return _fail(return_indices)
#         except Exception:
#             return _fail(return_indices)

#     # extract subimage safely
#     try:
#         subimage = image[xstart:xend, ystart:yend]
#         # if subimage is empty, fail
#         if subimage.size == 0:
#             return _fail(return_indices)
#     except Exception:
#         return _fail(return_indices)

#     # optional immediate show of subimage (keep original behaviour)
#     if show:
#         try:
#             plt.figure()
#             plt.imshow(subimage)
#             plt.show()
#         except Exception:
#             pass

#     # now the numeric processing; any failure returns sentinel
#     try:
#         # compute ESF and LSF robustly
#         esf = np.mean(subimage, axis=axis)

#         # protect if esf has no length
#         if esf is None or getattr(esf, "size", 0) == 0:
#             return _fail(return_indices)

#         lsf = np.zeros(len(esf))
#         if len(esf) >= 3:
#             lsf[1:-1] = esf[2:] / 2.0 - esf[:-2] / 2.0
#         else:
#             # too small to compute lsf robustly
#             return _fail(return_indices)

#         # find lower-half and higher-half intersections as before,
#         # but guard every step that can return invalid indices.
#         # helper to call get_surrounding_indices safely
#         def _safe_get_surrounding(arr, target):
#             try:
#                 return get_surrounding_indices(arr, target, ignore_warning)
#             except Exception:
#                 return -1, -1

#         half_max_low = np.min(lsf) / 2.0
#         min_idx_arr = np.where(lsf == np.min(lsf))[0]
#         if min_idx_arr.size == 0:
#             return _fail(return_indices)
#         min_index = int(min_idx_arr[0])

#         idx0, idx1 = _safe_get_surrounding(lsf[: min_index + 1], half_max_low)
#         if idx0 == -1:
#             return _fail(return_indices)
#         intersect0 = get_line_crossing((idx0, lsf[idx0]), (idx1, lsf[idx1]), half_max_low)[0]

#         idx0, idx1 = _safe_get_surrounding(lsf[min_index:], half_max_low)
#         if idx0 == -1:
#             return _fail(return_indices)
#         idx0 = int(idx0) + min_index
#         idx1 = int(idx1) + min_index
#         intersect1 = get_line_crossing((idx0, lsf[idx0]), (idx1, lsf[idx1]), half_max_low)[0]

#         half_max_high = np.max(lsf) / 2.0
#         max_idx_arr = np.where(lsf == np.max(lsf))[0]
#         if max_idx_arr.size == 0:
#             return _fail(return_indices)
#         max_index = int(max_idx_arr[0])

#         idx0, idx1 = _safe_get_surrounding(lsf[: max_index + 1], half_max_high)
#         if idx0 == -1:
#             return _fail(return_indices)
#         intersect2 = get_line_crossing((idx0, lsf[idx0]), (idx1, lsf[idx1]), half_max_high)[0]

#         idx0, idx1 = _safe_get_surrounding(lsf[max_index:], half_max_high)
#         if idx0 == -1:
#             return _fail(return_indices)
#         idx0 = int(idx0) + max_index
#         idx1 = int(idx1) + max_index
#         intersect3 = get_line_crossing((idx0, lsf[idx0]), (idx1, lsf[idx1]), half_max_high)[0]

#     except Exception:
#         # Any numeric error should not crash the pipeline.
#         # Return sentinel to let calling code handle the failure.
#         return _fail(return_indices)

#     # optional plotting of esf/lsf (same as original)
#     if show:
#         try:
#             plt.figure()
#             plt.plot(esf - np.mean(esf), ".-", label="ESF")
#             plt.plot(lsf, ".-", label="LSF")
#             plt.legend()
#             plt.title("ESF and LSF")
#             plt.xlabel("Pixel Index")
#             plt.axhline(y=np.max(lsf) / 2, color="r", linestyle="--")
#             plt.axhline(y=np.min(lsf) / 2, color="r", linestyle="--")
#             plt.axvline(x=intersect0, color="b", linestyle="--")
#             plt.axvline(x=intersect1, color="b", linestyle="--")
#             plt.axvline(x=intersect2, color="b", linestyle="--")
#             plt.axvline(x=intersect3, color="b", linestyle="--")
#             plt.show()
#         except Exception:
#             pass

#     # return with same shapes as original
#     if return_indices:
#         return lsf, esf, xstart, xend, ystart, yend, subimage
#     return lsf, esf, subimage

def get_lsf_esf(
    image,
    center_point,
    axis=0,
    avg_num=101,
    plot_range=75,
    show=True,
    return_indices=False,
    ignore_warning=False,
):
    # might eventually want to interpolate more

    def fail():
        if return_indices:
            return (-1, -1, -1, -1, -1, -1, -1)
        else:
            return -1, -1, -1

    # what if we determine that the line is too close to the edge of the image?
    min_distance = 10
    if center_point[1] < min_distance or center_point[0] < min_distance:
        return fail()
    if (image.shape[1] - center_point[0]) < min_distance:
        return fail()
    if (image.shape[0] - center_point[1]) < min_distance:
        return fail()

    if axis == 0:
        xstart = max(int(center_point[1] - avg_num / 2), 0)
        xend = min(int(center_point[1] + avg_num / 2), image.shape[0])
        ystart = max(int(center_point[0] - plot_range / 2), 0)
        yend = min(int(center_point[0] + plot_range / 2), image.shape[1])
    elif axis == 1:
        xstart = max(int(center_point[1] - plot_range / 2), 0)
        xend = min(int(center_point[1] + plot_range / 2), image.shape[0])
        ystart = max(int(center_point[0] - avg_num / 2), 0)
        yend = min(int(center_point[0] + avg_num / 2), image.shape[1])
    else:
        raise ValueError(f"axis must be 0 or 1, not {axis}")
    subimage = image[xstart:xend, ystart:yend]

    def fail():
        if return_indices:
            return (-1, -1, -1, -1, -1, -1, subimage)
        else:
            return -1, -1, subimage

    if show:
        plt.figure()
        plt.imshow(subimage)
        plt.show()

    esf = np.mean(subimage, axis=axis)

    # 2023/02/09 trying new way of calculating lsf
    lsf = np.zeros(len(esf))
    lsf[1:-1] = esf[2:] / 2 - esf[:-2] / 2

    # find first intersection point for lower half max
    half_max = np.min(lsf) / 2
    min_index = np.where(lsf == np.min(lsf))[0][0]
    idx0, idx1 = get_surrounding_indices(lsf[: min_index + 1], half_max, ignore_warning)
    if idx0 == -1:
        return fail()
    intersect0 = get_line_crossing((idx0, lsf[idx0]), (idx1, lsf[idx1]), half_max)[0]

    # find second intersection point for lower half max
    idx0, idx1 = get_surrounding_indices(lsf[min_index:], half_max, ignore_warning)
    if idx0 == -1:
        return fail()
    idx0 = idx0 + min_index
    idx1 = idx1 + min_index
    intersect1 = get_line_crossing((idx0, lsf[idx0]), (idx1, lsf[idx1]), half_max)[0]

    # find first intersection point for higher half max
    half_max = np.max(lsf) / 2
    max_index = np.where(lsf == np.max(lsf))[0][0]
    idx0, idx1 = get_surrounding_indices(lsf[: max_index + 1], half_max, ignore_warning)
    if idx0 == -1:
        return fail()
    intersect2 = get_line_crossing((idx0, lsf[idx0]), (idx1, lsf[idx1]), half_max)[0]

    # find second intersection point for higher half max
    idx0, idx1 = get_surrounding_indices(lsf[max_index:], half_max, ignore_warning)
    if idx0 == -1:
        return fail()
    idx0 = idx0 + max_index
    idx1 = idx1 + max_index
    intersect3 = get_line_crossing((idx0, lsf[idx0]), (idx1, lsf[idx1]), half_max)[0]

    if show:
        plt.figure()
        plt.plot(
            esf - np.mean(esf),
            ".-",
            label="ESF",
        )
        plt.plot(
            lsf,
            ".-",
            label="LSF",
        )
        plt.legend()
        plt.title("ESF and LSF")
        plt.xlabel("Pixel Index")
        plt.axhline(y=np.max(lsf) / 2, color="r", linestyle="--")
        plt.axhline(y=np.min(lsf) / 2, color="r", linestyle="--")
        plt.axvline(x=intersect0, color="b", linestyle="--")
        plt.axvline(x=intersect1, color="b", linestyle="--")
        plt.axvline(x=intersect2, color="b", linestyle="--")
        plt.axvline(x=intersect3, color="b", linestyle="--")

        plt.show()

    if return_indices:
        return lsf, esf, xstart, xend, ystart, yend, subimage
    return lsf, esf, subimage


# take information about a point in an image
# and re-center it around a detected line
# this is an improvement on a previous function
# which we might want to go back and change
# TODO: don't always use default lsf_range
def adjust_center(image, binary_image, center_point, axis, debug=False, lsf_range=150): 
    # Changed lsf_range from 150 to new value- Abhinav 2/20
    margin = int(lsf_range / 2)

    if (
        center_point[0] < margin or
        center_point[0] > image.shape[1] - margin or
        center_point[1] < margin or
        center_point[1] > image.shape[0] - margin
    ):
        return (-1, -1)

    # assess whether there is actually a line here
    lsf, esf, xstart, xend, ystart, yend, sub_image = get_lsf_esf(
        image,
        center_point,
        axis=axis,
        return_indices=True,
        show=debug,
        plot_range=lsf_range,
        avg_num=lsf_range,  # TODO: think about this more?
        ignore_warning=True,
    )

    if isinstance(esf, int) and esf == -1:
        if debug:
            print("failure to find esf/lsf on raw image")
        return (-1, -1)

    # assess whether there is actually a line here
    # TODO: switch "show" back to False
    lsf2, esf2, xstart2, xend2, ystart2, yend2, sub_image = get_lsf_esf(
        binary_image,
        center_point,
        axis=axis,
        return_indices=True,
        show=debug,
        plot_range=lsf_range,
        ignore_warning=True,
    )

    if isinstance(esf2, int) and esf2 == -1:
        return (-1, -1)

    # not totally sure if this is best choice for crop_distance
    valid = determine_line_validity(
        binary_image,
        crop_distance=int(lsf_range / 2),
        center_point=center_point,
        esf=esf2,
        lsf=lsf2,
        show=debug,
        show_if_failed=debug,
    )
    if not valid:
        # determine_line_validity(binary_image, center_point, axis, esf2, lsf2, show=True)
        return (-1, -1)

    # this is failing when there's a significant tilt in intensity
    # I wonder if we can... maybe filter and just subtract it out
    # TODO: clean this up later
    num = int(len(esf) / 4)
    l0 = np.mean(esf[:num])
    l1 = np.mean(esf[-num:])
    line = np.linspace(l0, l1, len(esf))
    esf_ = esf - (line - (l0 + l1) / 2)
    mid_index = np.where(esf_ == np.min(esf_))[0][0]

    indexR = mid_index - 1
    while lsf[indexR] < 0:
        indexR = indexR + 1
    indexL = indexR - 1
    midpoint = get_line_crossing((indexR, lsf[indexR]), (indexL, lsf[indexL]), 0)

    if debug:
        plt.figure()
        plt.plot(lsf, ".-", label="lsf")
        plt.plot(np.zeros(len(lsf)))
        plt.plot(esf - np.mean(esf), ".-", label="esf")
        plt.axvline(x=midpoint[0], label="midpoint")
        plt.legend()
        plt.title(f"Adjusting point: {center_point} \n {midpoint[0]}")

    if axis == 1:
        return (center_point[0], xstart + midpoint[0])
    else:
        return (ystart + midpoint[0], center_point[1])


# take a bunch of approximate points in an image
# and return the location of graph vertices
def get_all_vertices(
    image,
    approx_points,
    binary_threshold_values,
    debug=False,
    show=False,
    lsf_range=75,
    display_downsample=1,
):
    binary_image = make_binary_image(image, binary_threshold_values)

    # adjust center in both directions
    vertices = []
    for point in approx_points:
        dir1_adj = adjust_center(
            image, binary_image, point, axis=1, lsf_range=lsf_range, debug=debug
        )
        dir0_adj = adjust_center(
            image, binary_image, point, axis=0, lsf_range=lsf_range, debug=debug
        )

        if -1 in dir0_adj or -1 in dir1_adj:
            continue

        vertices.append([dir0_adj[0], dir1_adj[1]])

    if show:
        display_with_points(image, vertices, display_downsample=display_downsample)
    print("Approx points:", len(approx_points))
    print("Final vertices:", len(vertices))

    return vertices
