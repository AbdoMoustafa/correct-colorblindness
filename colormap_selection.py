import numpy as np
from colorama import Fore, Style
from colormath.color_objects import sRGBColor, LabColor
from colormath.color_conversions import convert_color
from colormath.color_diff import delta_e_cmc
from PIL import Image
from daltonlens import simulate
import cv2
import math


def compute_color_distance_matrix(colors):
    """
    Compute color distance matrix to observe similarity between colors
    """
    num_colors = len(colors)
    color_distance_matrix = np.zeros((num_colors, num_colors))

    for i in range(num_colors):
        for j in range(i + 1, num_colors):
            diff = color_difference(colors[i], colors[j])
            color_distance_matrix[i, j] = diff
            color_distance_matrix[j, i] = diff  # Symmetric

    return color_distance_matrix


def color_difference(color1, color2):
    """
    Calculate the deltaE (CMC l:c) between two colors.
    """
    color1_rgb = sRGBColor(*color1)
    color2_rgb = sRGBColor(*color2)

    color1_lab = convert_color(color1_rgb, LabColor)
    color2_lab = convert_color(color2_rgb, LabColor)

    # Divide by 2 because formula isn't symmetric
    return (delta_e_cmc(color1_lab, color2_lab) + delta_e_cmc(color2_lab, color1_lab)) / 2.0


def simulate_colorblindness(color, deficiency='tritan', severity=1):
    """
    Simulate color blindness on a given color using the DaltonLens library.
    """

    image = Image.fromarray(np.array([[color]], dtype=np.uint8))

    # Brettel1997  Simulator
    simulator = simulate.Simulator_Brettel1997()

    # Mapping deficiency names
    deficiency_mapping = {
        'protan': simulate.Deficiency.PROTAN,
        'deutan': simulate.Deficiency.DEUTAN,
        'tritan': simulate.Deficiency.TRITAN
    }

    # Simulating color blindness on the image
    simulated_im = simulator.simulate_cvd(np.asarray(image), deficiency_mapping[deficiency], severity=severity)

    # Extracting simulated color from the image
    simulated_color = tuple(simulated_im[0, 0])

    return simulated_color


def valid(color):
    """
    Filter white
    """

    return not all(c > 245 for c in color)


def select_colors(cmap, colors):
    """
    Selecting qualitative color palette, ensure max 10
    """

    if colors <= 10:
        palette = cmap[0]
    else:
        raise ValueError(
            Fore.LIGHTYELLOW_EX + Style.BRIGHT + "We only accept a qualitative plots with a maximum of 10 colors")

    return palette[1][:colors]


def evaluate_colormap(colormap, name, original_ratio):
    """
    Evaluate colormap under simulated conditions
    """
    normal_distances = compute_color_distance_matrix(colormap)
    colorblind_distances = compute_color_distance_matrix([simulate_colorblindness(color) for color in colormap])
    ratios = compute_color_distance_ratios(normal_distances, colorblind_distances)
    old_avg = np.mean(original_ratio)
    new_avg = np.mean(ratios)

    percentage_improvement = ((old_avg - new_avg) / old_avg) * 100

    score = evaluate_ratios(ratios)
    return score, percentage_improvement


def compute_color_distance_ratios(normal_distances, colorblind_distances):
    """
    Ratio matrix to get ratio of normal distance matrix to colorblind distance matrix
    """
    colorblind_distances_copy = colorblind_distances.copy()

    colorblind_distances_copy[colorblind_distances_copy == 0] = 1e-10
    ratios_matrix = normal_distances / colorblind_distances_copy

    return ratios_matrix


def get_weight(ratio):
    """
    Adds higher weight to color pairs that are more difficult to distinguish
    """
    return math.exp(ratio - 1)  # Exponential


def evaluate_ratios(ratio_matrix):
    """
    Provide a weighted score to a colormap based on how distinguishable it is under colorblind conditions
    (Only used initially to determine the best map for each cvd)
    """
    penalty_score = 0

    # Iterate over each ratio in the matrix
    for i in range(ratio_matrix.shape[0]):
        for j in range(ratio_matrix.shape[1]):
            if i != j:  # Exclude the diagonal (ratios of a color with itself)
                ratio = ratio_matrix[i, j]

                # Calculate penalty based on the ratio and the corresponding weight
                weight = get_weight(ratio)
                penalty = weight * ratio

                # Add the penalty to the total score
                penalty_score += penalty

    return penalty_score


def generate_colormap_image(cmap, width=27):
    """
    Generating gradient with the desired colormap
    """
    # Generating a gradient
    gradient = np.linspace(0, 255, 256, dtype=np.uint8)

    gradient = np.tile(gradient, (width, 1))

    # Transposing the gradient to make it vertical
    gradient = np.transpose(gradient)

    # Applying the colormap
    if isinstance(cmap, int):  # Check if the colormap is from OpenCV
        colored_gradient = cv2.applyColorMap(gradient, cmap)
    else:  # Otherwise Matplotlib
        colored_gradient = (cmap(gradient / 255.0)[:, :, :3] * 255).astype(np.uint8)

    return colored_gradient
