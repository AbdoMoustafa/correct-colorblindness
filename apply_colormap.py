import numpy as np
from scipy.spatial import KDTree
from scipy.interpolate import PchipInterpolator
import cv2
from colormath.color_objects import sRGBColor, LabColor
from skimage.color import rgb2lab, lab2rgb
from colormath.color_conversions import convert_color
import colorsys


def create_color_mapping(input_colors, desired_colors):
    """
    Create color mapping for continuous plots
    """
    # Polynomial Interpolation
    interpolated_desired_colors = interpolate_colors(np.array(desired_colors), len(input_colors))

    # Map each input color to the corresponding interpolated desired color
    mapping = {tuple(input_color): tuple(interpolated_desired_color)
               for input_color, interpolated_desired_color in zip(input_colors, interpolated_desired_colors)}

    return mapping


def create_color_mapping_discrete(input_colors, desired_colors):
    """
    Create color mapping for discrete plots
    """
    mapping = {}
    mapping = {original: new for original, new in zip(input_colors, desired_colors)}

    return mapping


def interpolate_colors(desired_colors_rgb, num_colors):
    """
     Polynomial Interpolation for continuous plots
    """

    # Convert RGB colors to LAB & set to range [0, 1]
    desired_colors_lab = rgb2lab(desired_colors_rgb / 255.0)

    original_positions = np.linspace(0, 1, len(desired_colors_lab))
    new_positions = np.linspace(0, 1, num_colors)

    # Interpolate each channel in LAB space
    interpolated_colors_lab = np.empty((num_colors, 3))
    for i in range(3):
        interpolator = PchipInterpolator(original_positions, desired_colors_lab[:, i])
        interpolated_colors_lab[:, i] = interpolator(new_positions)

    # Convert interpolated LAB colors back to RGB, and back to [0, 255]
    interpolated_colors_rgb = lab2rgb(interpolated_colors_lab) * 255.0

    return interpolated_colors_rgb


def apply_color_mapping(image, color_mapping, dimensions, non_heatmap, continuous=False):
    """
     Apply colormapping to plots
    """
    # Convert to RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Create a mask for color regions
    mask = np.zeros(image.shape[:2], dtype=bool)
    for (x1, y1, x2, y2) in dimensions:
        mask[y1:y2, x1:x2] = True

    # Invert for faster lookup
    inv_color_mapping = {v: k for k, v in color_mapping.items()}

    # Extract all pixels
    masked_pixels = image_rgb[mask]

    # Find unique colors
    unique_colors, return_inverse = np.unique(masked_pixels, axis=0, return_inverse=True)

    # Convert to LAB and k-d tree
    rgb_to_lab = {rgb: convert_color(sRGBColor(*rgb, is_upscaled=True), LabColor).get_value_tuple() for rgb in
                  color_mapping.keys()}
    lab_to_rgb = {lab: rgb for rgb, lab in rgb_to_lab.items()}
    color_tree = KDTree(list(lab_to_rgb.keys()))

    # Processing unique colors
    new_colors = np.zeros_like(unique_colors)
    for idx, color in enumerate(unique_colors):
        original_color = tuple(color)

        # if color already in the mapping, apply new color
        if original_color in inv_color_mapping:
            new_color = inv_color_mapping[original_color]
        # Else find closest color in mapping
        else:
            original_lab = rgb_to_lab.get(original_color, convert_color(sRGBColor(*original_color, is_upscaled=True),
                                                                        LabColor).get_value_tuple())
            dist, index = color_tree.query(original_lab)
            nearest_lab_color = color_tree.data[index]
            nearest_rgb_color = lab_to_rgb[tuple(nearest_lab_color)]
            new_color = color_mapping.get(nearest_rgb_color, nearest_rgb_color)

            new_color = adjust_color(original_color, nearest_rgb_color, new_color)

        new_colors[idx] = new_color

    # Map the new colors back to the original image
    image_rgb[mask] = new_colors[return_inverse].reshape(image_rgb[mask].shape)

    # Convert back to BGR
    final_image = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    return final_image


def is_pixel_in_segments(x, y, segments):
    """
    Check to see if a pixel is within certain dimensions
    """
    for x1, y1, x2, y2 in segments:
        if x1 <= x < x2 and y1 <= y < y2:
            return True
    return False


def adjust_color(original_color, base_color, new_color):
    """
    Adjust discrete colors in HLS
    """
    # Convert RGB colors to a range of [0, 1]
    original_color = [c / 255.0 for c in original_color]
    base_color = [c / 255.0 for c in base_color]
    new_color = [c / 255.0 for c in new_color]

    # Avoid divison by 0
    def safe_hls_conversion(rgb):
        max_comp = max(rgb)
        min_comp = min(rgb)
        if max_comp + min_comp == 2.0:

            return (0, rgb[0], 0)  # Hue and saturation are irrelevant for grey
        else:
            return colorsys.rgb_to_hls(*rgb)

    original_hls = safe_hls_conversion(original_color)
    base_hls = safe_hls_conversion(base_color)
    new_hls = safe_hls_conversion(new_color)

    # get lightness and saturation differences
    lightness_diff = original_hls[1] - base_hls[1]
    saturation_diff = original_hls[2] - base_hls[2]

    # Adjust lightness and saturation
    adjusted_hls = (
        new_hls[0],
        max(0, min(1, new_hls[1] + lightness_diff)),
        max(0, min(1, new_hls[2] + saturation_diff))
    )

    # Convert back to RGB
    adjusted_rgb = colorsys.hls_to_rgb(*adjusted_hls)
    adjusted_rgb = [int(c * 255) for c in adjusted_rgb]

    return adjusted_rgb


def interpolate_color_discrete(original_color, mean_color, target_color, amplification_factor=1.2):
    """
    Interpolate lightness for discrete colors to mitigate anti-aliasing affect
    """
    # Convert to Lab color space
    original_lab = convert_color(sRGBColor(*original_color, is_upscaled=True), LabColor)
    mean_lab = convert_color(sRGBColor(*mean_color, is_upscaled=True), LabColor)
    target_lab = convert_color(sRGBColor(*target_color, is_upscaled=True), LabColor)

    # Calculate the difference in lightness between the original and the main color
    lightness_diff = (original_lab.lab_l - mean_lab.lab_l) * amplification_factor

    # Interpolate around the target color and white/black based on lightness
    new_lightness = max(0, min(100, target_lab.lab_l + lightness_diff))

    # Interpolated Lab color with adjusted lightness
    interpolated_lab = LabColor(
        lab_l=new_lightness,
        lab_a=target_lab.lab_a,
        lab_b=target_lab.lab_b
    )

    # Convert back to RGB
    interpolated_rgb = convert_color(interpolated_lab, sRGBColor).get_upscaled_value_tuple()
    return interpolated_rgb