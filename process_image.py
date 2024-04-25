import numpy as np
import cv2
import matplotlib.cm
import os
from colorama import Fore, Back, Style

import apply_colormap
import color_extraction
import refinements
import gridlines
import colormap_selection

"""
Colormaps
RG - Red/Green 
BY - Blue/Yellow
"""
DIVERGING_COLOR_MAPS_RG = [
    ("PuOr", matplotlib.cm.PuOr)
]

DIVERGING_COLOR_MAPS_BY = [
    ("RdGy", matplotlib.cm.RdGy)
]

SEQUENTIAL_MAPS_RG = [
    ("CIVIDIS", matplotlib.cm.cividis)
]

SEQUENTIAL_MAPS_BY = [
    ("Reds", matplotlib.cm.Reds)
]

QUALITATIVE_COLOR_MAPS_RG = [
    ("10 Colours",
     [(188, 195, 195), (63, 144, 218), (255, 169, 14), (185, 172, 112), (130, 150, 190), (50, 50, 240), (255, 230, 150),
      (0, 102, 102), (103, 107, 119), (146, 218, 221)])
]

QUALITATIVE_COLOR_MAPS_BY = [
    ("10 Colours BY",
     [(188, 195, 195), (0, 0, 128), (220, 20, 60), (34, 139, 34),
      (210, 105, 30), (186, 85, 211), (255, 99, 71),
      (112, 128, 144), (0, 191, 255), (255, 165, 0)])
]


def process_image(input_image_path, results_folder, deficiency, user_choice):
    continuous = False
    clash = True
    image = cv2.imread(input_image_path)

    # Dimensions of color regions
    _, _, extended_dimensions = color_extraction.find_color_regions(image)

    non_heatmap = False
    image = cv2.imread(input_image_path)

    try:
        legend_type, box = color_extraction.detect_legend(input_image_path)
    except Exception as e:
        print("Failed to detect legend")
    if legend_type == 'Legend':
        non_heatmap = True
    if legend_type == 'Monochrome':
        print("This image is already colorblind safe!")

    legend = None

    if not non_heatmap:
        try:
            legend, legend_dimensions = color_extraction.isolate_legend_heatmap(input_image_path, box)

            legend_x1, legend_y1, legend_x2, legend_y2 = legend_dimensions
        except Exception as e:
            print("ERROR: There has been a problem isolating the legend")

        try:
            heatmap = color_extraction.detect_heatmap(input_image_path)
            dimensions = tuple([int(x) for x in heatmap])
            x1, y1, x2, y2 = dimensions

            # Refine bounding box for greater accuracy
            dimensions = refinements.adjust_bounding_box(image, (x1, y1, x2, y2))
            grid_dimensions = dimensions
            num_x1, num_y1, num_x2, num_y2 = grid_dimensions
            cropped_image = image[num_y1:num_y2, num_x1:num_x2]
            cv2.imwrite("cropped_heatmap.png", cropped_image)

            # Extract numbers (if present)
            numbers = gridlines.ocr("cropped_heatmap.png")
            numbers = gridlines.preprocess_dict(numbers)

            grid_dimensions = (num_x1, num_y1, num_x2, num_y2)
            clash = False
            offset = (num_x1, num_y1)
            numbers = gridlines.adjust_offset(numbers, offset)

            filtered_numbers = {}
            dropped_numbers = {}

            # Check for Google API misdetections
            for k, v in numbers.items():
                if gridlines.are_all_numbers(v):
                    filtered_numbers[k] = v
                else:
                    dropped_numbers[k] = v

            # If elements were dropped, Google Vision API made misdetections due to clashes with other elements
            if len(dropped_numbers) > 0:
                clash = True
            elif len(numbers) > 0:
                image = gridlines.remove_numbers(numbers, image)

        except Exception as e:
            continuous = True
            dimensions = color_extraction.find_color_regions(image)[2]
            print("There has been a problem detecting the heatmap")

        legend_dimensionsIMG = refinements.adjust_bounding_box(image,
                                                               (legend_x1, legend_y1, legend_x2, legend_y2))

        legend_dimensions, legend, dimensions = color_extraction.rotate_legend(legend_dimensionsIMG, legend,
                                                                               dimensions)

    diverging, qualatative, sequential, non_heatmap = color_extraction.get_distribution(legend_type)

    if legend is not None:
        colors = color_extraction.extract_sample_colors_from_legend(legend)

        if qualatative:
            all_colors = color_extraction.extract_frequent_colors_from_legend(legend)
            colors = all_colors

        else:
            all_colors = color_extraction.extract_all_colors_from_legend(legend)

    else:
        non_heatmap = True
        qualatative = True

        legend = image[box[1]:box[3], box[0]:box[2]]

        color_dict, all_colors, drawn_image = color_extraction.extract_all_colors_discrete(legend)
        colors = all_colors

    color_distance_matrix = colormap_selection.compute_color_distance_matrix(colors)

    # Simulate colorblindness for each color in the palette
    colorblind_colors = [colormap_selection.simulate_colorblindness(color, deficiency=deficiency) for color in colors]
    colorblind_matrix = colormap_selection.compute_color_distance_matrix(colorblind_colors)

    # Compute matrix of ratio between two matrices
    ratios_matrix = colormap_selection.compute_color_distance_ratios(color_distance_matrix, colorblind_matrix)
    average = np.mean(ratios_matrix)

    # If input is less that 10% less distinguishable to cvd vs normal vision, it's considered colorblind safe
    if average < 1.1:
        print(
            Back.GREEN + Style.BRIGHT + " COMPLETE " + Back.RESET + Fore.GREEN + f"\n\nThis image is already colorblind safe for your CVD!")
        base_name = os.path.splitext(os.path.basename(input_image_path))[0]
        output_file = os.path.join(results_folder, f"{base_name}.png")
        cv2.imwrite(output_file, image)
        return

    min_score = float('inf')
    best_cmap_name = None
    percentage_improvement = 0

    if qualatative or non_heatmap:
        if deficiency == 'tritan':
            distribution = QUALITATIVE_COLOR_MAPS_BY
        else:
            distribution = QUALITATIVE_COLOR_MAPS_RG
    elif diverging:
        if deficiency == 'tritan':
            distribution = DIVERGING_COLOR_MAPS_BY
        else:
            distribution = DIVERGING_COLOR_MAPS_RG
    elif deficiency == 'tritan':
        distribution = SEQUENTIAL_MAPS_BY
    else:
        distribution = SEQUENTIAL_MAPS_RG

    if qualatative:
        all_cmap_colors = colormap_selection.select_colors(distribution, len(all_colors))

        new_color_distance_matrix = colormap_selection.compute_color_distance_matrix(colors)
        new_colorblind_colors = [colormap_selection.simulate_colorblindness(color, deficiency=deficiency) for color in
                                 all_cmap_colors]
        new_colorblind_matrix = colormap_selection.compute_color_distance_matrix(new_colorblind_colors)
        new_ratios_matrix = colormap_selection.compute_color_distance_ratios(new_color_distance_matrix,
                                                                             new_colorblind_matrix)
        new_average = np.mean(new_ratios_matrix)
        percentage_improvement = ((average - new_average) / average) * 100

    filtered_distribution = []
    for name, cmap in distribution:

        if not non_heatmap and not qualatative:

            cmap_legend = colormap_selection.generate_colormap_image(cmap)

            cmap_colors = color_extraction.extract_sample_colors_from_legend(cmap_legend)

            score, improvement = colormap_selection.evaluate_colormap(cmap_colors, name, ratios_matrix)

            if score < min_score:  # Checking if this score is the lowest so far
                min_score = score
                best_cmap_name = name
                percentage_improvement = improvement

                if non_heatmap:

                    all_cmap_colors = cmap_colors
                    # all_cmap_colors = sort_colors_by_saturation(cmap_colors)
                else:
                    all_cmap_colors = color_extraction.extract_all_colors_from_legend(cmap_legend)

    # print(f"The best colormap is {best_cmap_name} with a score of {min_score}")
    if not non_heatmap and not qualatative:
        color_mapping = apply_colormap.create_color_mapping(all_colors, all_cmap_colors)
    else:
        color_mapping = apply_colormap.create_color_mapping_discrete(all_colors, all_cmap_colors)

    if non_heatmap:

        for mean_color, associated_colors in color_dict.items():
            # Retrieve the corresponding new color from color_mapping for this mean color
            new_color = color_mapping[mean_color]

            # Add each associated color as a key in color_mapping with an interpolated mapped value
            for original_color in associated_colors:
                # Interpolate color based on the original color's lightness compared to the mean color
                interpolated_color = apply_colormap.interpolate_color_discrete(original_color, mean_color,
                                                                               new_color)
                color_mapping[original_color] = interpolated_color

    if non_heatmap or continuous:
        color_mapping[(255, 255, 255)] = (255, 255, 255)
        color_mapping[(0, 0, 0)] = (0, 0, 0)

    if non_heatmap:
        color_mapping[(0, 0, 0)] = (0, 0, 0)
        image = apply_colormap.apply_color_mapping(image, color_mapping, extended_dimensions, non_heatmap, continuous)
        image = refinements.apply_light_gaussian_blur(image)

    else:
        image = apply_colormap.apply_color_mapping(image, color_mapping, dimensions, non_heatmap, continuous)

        if gridlines.detect_grid(input_image_path) and not continuous:
            image = gridlines.grid_lines(cv2.imread(input_image_path), image, grid_dimensions)

        if not clash:
            image = gridlines.draw_text_in_boxes(image, numbers)

    base_name = os.path.splitext(os.path.basename(input_image_path))[0]
    output_file = os.path.join(results_folder, f"{base_name}_colormapped.png")
    cv2.imwrite(output_file, image)

    if user_choice == '1':
        print(
            Back.GREEN + Style.BRIGHT + " COMPLETE " + Back.RESET + Fore.GREEN + f"\n\nYour new image has been processed and saved to: {output_file}")
    else:
        print(Back.GREEN + Style.BRIGHT + " COMPLETE ")
