from ultralytics import YOLO
import cv2
import numpy as np


def detect_legend(input_image_path: str):
    """
    Method to detect the legend in a plot
    """
    model = YOLO("Object Detection Models/best.pt")
    results = model(input_image_path)

    confidences = results[0].boxes.conf
    max_conf_index = confidences.argmax().item()
    highest_conf_box = results[0].boxes.xyxy[max_conf_index].tolist()

    box = [int(x) for x in highest_conf_box]
    legend_type = results[0].names[int(results[0].boxes.cls[max_conf_index])]

    return legend_type, box


def detect_heatmap(input_image_path: str):
    """
    Method to detect the heatmap
    """
    model = YOLO("Object Detection Models/heatmap_detect.pt")
    heatmap = model(input_image_path)
    confidences = heatmap[0].boxes.conf
    if (max(confidences)) < 0.75:
        return None
    max_conf_index = confidences.argmax().item()
    heatmap = heatmap[0].boxes.xyxy[max_conf_index].tolist()
    return heatmap


def find_color_regions(image):
    """
    Method to detect regions of color in discrete plots
    """
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Tresholds for color detection
    lower_color = np.array([0, 30, 30])
    upper_color = np.array([180, 255, 255])
    color_mask = cv2.inRange(hsv_image, lower_color, upper_color)

    contours, _ = cv2.findContours(color_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    kernel = np.ones((3, 3), np.uint8)
    contour_image = np.copy(image_rgb)
    bounding_boxes = []
    cv2.imwrite('color_mask.png', color_mask)

    # Processing each contour
    for contour in contours:

        x, y, w, h = cv2.boundingRect(contour)
        bounding_boxes.append((x, y, w, h))

        mask = np.zeros_like(color_mask)
        cv2.drawContours(mask, [contour], -1, 255, -1)

        # Center of each contour
        M = cv2.moments(contour)
        if M['m00'] == 0:  # Avoid division by zero
            continue
        cX = int(M['m10'] / M['m00'])
        cY = int(M['m01'] / M['m00'])

        # Getting center color of each contour
        size = 3
        center_color = cv2.mean(image_rgb[cY - size:cY + size, cX - size:cX + size])

        dilated_mask = cv2.dilate(mask, kernel, iterations=1)
        image_rgb[np.where(dilated_mask == 255)] = center_color[:3]

        # Drawing contiurs (for testing)
        cv2.drawContours(contour_image, [contour], -1, (255, 0, 0), 1)

    corrected_image = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    contour_image_bgr = cv2.cvtColor(contour_image, cv2.COLOR_RGB2BGR)

    extended_dimensions = []

    # calculating x2 and y2 for each bounding box
    for (x1, y1, width, height) in bounding_boxes:
        x2 = x1 + width
        y2 = y1 + height
        extended_dimensions.append((x1, y1, x2, y2))

    return corrected_image, contour_image_bgr, extended_dimensions


def isolate_legend_heatmap(input_image_path, dimensions):
    """
    Isolating detected legend
    """
    image = cv2.imread(input_image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    x1, y1, x2, y2 = [int(x) for x in dimensions]

    legend = image_rgb[y1:y2, x1:x2]

    # Convert to numpy array
    legend_array = np.array(legend)

    width, height = x2 - x1, y2 - y1
    legend_dimensions = (x1, y1, width, height)
    legend_x1, legend_y1, width, height = legend_dimensions
    legend_x2, legend_y2 = legend_x1 + width, legend_y1 + height

    return legend_array, (legend_x1, legend_y1, legend_x2, legend_y2)


def rotate_legend(legend_dimensions, legend, dimensions):
    """
    Method to rotate horizontal legend in heatmap (if detected)
    """
    legend_x1, legend_y1, legend_x2, legend_y2 = legend_dimensions

    if (legend_x2 - legend_x1) > (legend_y2 - legend_y1):

        # If legend is more wide than tall, it's likely horizontal
        legend = rotate_image(legend)

        # Update dimensions
        legend_dimensions_new = (legend_y1, legend_x1, legend_y2 - legend_y1, legend_x2 - legend_x1)
    else:
        # No rotation needed if vertical
        legend_dimensions_new = (legend_x1, legend_y1, legend_x2 - legend_x1, legend_y2 - legend_y1)

    if not isinstance(dimensions, list):
        dimensions = [dimensions]
    dimensions.append(tuple(legend_dimensions))
    return legend_dimensions_new, legend, dimensions


def rotate_image(image):
    """
    Rotates an image by 90 degrees.
    """
    # Rotate the image 90 degrees counterclockwise
    rotated_image = np.rot90(image)
    return rotated_image


def get_distribution(legend_type):
    """
    Sets the boolean variables of distributions based on the detected distribution by our model
    """
    diverging = False
    qualitative = False
    sequential = False
    non_heatmap = False

    if legend_type == "Diverging":
        diverging = True
    elif legend_type == "Qualitative":
        qualitative = True
    elif legend_type == "Sequential":
        sequential = True
    elif legend_type == "Legend":
        non_heatmap = True

    return diverging, qualitative, sequential, non_heatmap


def extract_sample_colors_from_legend(legend, num_colors=20, central_fraction=0.5):
    """
    Evenly select 20 colors along the color bar to compare with other
    """

    gray_legend = cv2.cvtColor(legend, cv2.COLOR_RGB2GRAY)
    _, thresh = cv2.threshold(gray_legend, 200, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2:]

    colors = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        center_x = x + w // 2
        center_y = y + h // 2

        # Extracting colors from a central region
        central_x1 = max(x, center_x - int(w * central_fraction / 2))
        central_x2 = min(x + w, center_x + int(w * central_fraction / 2))
        central_y1 = max(y, center_y - int(h * central_fraction / 2))
        central_y2 = min(y + h, center_y + int(h * central_fraction / 2))

        color_patch = legend[central_y1:central_y2, central_x1:central_x2]

        # Check if the color slice is not empty
        if color_patch.size > 0:
            color = np.mean(color_patch, axis=(0, 1))
            colors.append(tuple(map(int, color)))
        else:
            continue

    if len(colors) < num_colors:
        colors += [tuple(map(int, legend[int(i * legend.shape[0] / num_colors), legend.shape[1] // 2]))
                   for i in range(len(colors), num_colors)]

    return colors


def extract_all_colors_from_legend(legend):
    """
    Extracting all colors from the colorbar for colormapping
    """

    # Assume the gradient is vertical (we previously rotated horizontal ones)
    center_column = legend.shape[1] // 2

    # Trim
    trim_size = int(0.003 * legend.shape[0])
    unique_colors = []

    for i in range(trim_size, legend.shape[0] - trim_size):
        color = legend[i, center_column]
        unique_colors.append(tuple(map(int, color)))

    # Remove duplicates
    unique_colors = [unique_colors[i] for i in range(len(unique_colors)) if
                     i == 0 or unique_colors[i] != unique_colors[i - 1]]

    return unique_colors


def extract_frequent_colors_from_legend(legend, occurrence_threshold=5):
    """
    Extracting all colors from a qualitative colorbar for colormapping
    """

    center_column = legend.shape[1] // 2

    trim_size = int(0.01 * legend.shape[0])
    color_counts = {}

    for i in range(trim_size, legend.shape[0] - trim_size):
        color = tuple(legend[i, center_column])

        if color in color_counts:
            color_counts[color] += 1
        else:
            color_counts[color] = 1

    # Filter colors that occur at least 'occurrence_threshold' times
    frequent_colors = [color for color, count in color_counts.items() if count >= occurrence_threshold]

    return frequent_colors


def exact_center_color(image, x, y, w, h):
    """
    Calculate the color of the exact center pixel of a given bounding box.
    """

    center_x = x + w // 2
    center_y = y + h // 2
    center_color = image[center_y, center_x]
    return center_color


def extract_all_colors_discrete(image, x_tolerance=8, size_tolerance=20):
    """
    Extracting all colors from a discrete legend box
    """
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    lower_color = np.array([0, 30, 30])
    upper_color = np.array([180, 255, 255])
    color_mask = cv2.inRange(hsv_image, lower_color, upper_color)

    color_contours, _ = cv2.findContours(color_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    drawn_image = image.copy()
    legend_entries = []

    inner_boxes = []

    for c_col in color_contours:
        x_col, y_col, w_col, h_col = cv2.boundingRect(c_col)

        mean_color = cv2.mean(image[y_col:y_col + h_col, x_col:x_col + w_col])[:3]
        color_center = exact_center_color(image, x_col, y_col, w_col, h_col)
        color_center = (color_center[0], color_center[1], color_center[2])

        # Adding center colors of each detected contour
        if not all(value > 245 for value in mean_color):
            inner_boxes.append(
                (x_col, y_col, w_col, h_col, color_center))

    # Assume if multiple boxes in the legend overlap, they belong to the same data point
    if inner_boxes:
        inner_boxes = merge_close_boxes(inner_boxes, y_tolerance=5)

    aligned_boxes = []

    for box in inner_boxes:
        # Check x alignment and size similarity of contours as they should be similar
        if all(abs(box[0] - ib[0]) <= x_tolerance and
               abs(box[2] - ib[2]) <= size_tolerance and
               abs(box[3] - ib[3]) <= size_tolerance for ib in inner_boxes if ib is not box):
            aligned_boxes.append(box)

    x_col, y_col, w_col, h_col, color = box
    cv2.rectangle(drawn_image, (x_col, y_col), (x_col + w_col, y_col + h_col), (0, 255, 0), 2)

    legend_entries.append({'legend_color': color, 'bounding_box': (x_col, y_col, w_col, h_col)})

    colors = [color[4] for color in inner_boxes]
    color_dict = {}

    # Iterate over both the colors list and the aligned_boxes
    for color, box in zip(colors, aligned_boxes):
        x_col, y_col, w_col, h_col, _ = box

        # Extract colors from bounding box
        bounding_box_colors = extract_colors_from_bounding_box(image, (x_col, y_col, w_col, h_col))
        # Filter out white
        valid_colors = [col for col in bounding_box_colors if is_color_valid(col)]
        color_dict[color] = valid_colors

    return color_dict, colors, _


def merge_close_boxes(bounding_boxes, y_tolerance=1):
    """
    Merge close bounding boxes
    """

    bounding_boxes.sort(key=lambda x: x[1] + x[3] // 2)
    merged_boxes = []
    group = [bounding_boxes[0]]

    for i in range(1, len(bounding_boxes)):
        # Compare the y center of the current box with the last box
        _, y1, _, h1, _ = group[-1]
        y_center1 = y1 + h1 // 2
        _, y2, _, h2, _ = bounding_boxes[i]
        y_center2 = y2 + h2 // 2

        if abs(y_center1 - y_center2) <= y_tolerance:
            group.append(bounding_boxes[i])
        else:
            merged_boxes.append(group)
            group = [bounding_boxes[i]]

    if group:
        merged_boxes.append(group)

    final_boxes = []
    for group in merged_boxes:
        x_coords = [box[0] for box in group]
        y_coords = [box[1] for box in group]
        w_coords = [box[2] for box in group]
        h_coords = [box[3] for box in group]
        min_x = min(x_coords)
        min_y = min(y_coords)
        max_x = max(x_coords) + w_coords[x_coords.index(max(x_coords))]
        max_y = max(y_coords) + h_coords[y_coords.index(max(y_coords))]
        mean_colors = [box[4] for box in group]
        avg_color = tuple(np.mean(mean_colors, axis=0).astype(int))
        final_boxes.append((min_x, min_y, max_x - min_x, max_y - min_y, avg_color))

    return final_boxes


def extract_colors_from_bounding_box(image, bounding_box):
    """
    Extract all unique colors from bounding box
    """
    unique_colors = set()
    x, y, w, h = bounding_box

    # Iterate through each pixel in the bounding box
    for i in range(y, y + h):
        for j in range(x, x + w):
            color = image[i, j]

            # Exclude black, white
            if (not np.allclose(color, [0, 0, 0], atol=20) and
                    not np.allclose(color, [255, 255, 255], atol=20)):
                unique_colors.add(tuple(map(int, color)))

    return list(unique_colors)


def is_color_valid(color):
    """
    Function to check if all color components are 250 or below
    """
    return any(component <= 250 for component in color)
