import cv2
from PIL import Image, ImageDraw, ImageFont
import numpy as np
from google.cloud import vision_v1 as vision
from typing import Sequence
import re
from ultralytics import YOLO


def grid_lines(original_image, image, dimensions):
    """
    Method to draw gridlines on image
    """
    x1, y1, x2, y2 = dimensions

    # Draw vertical lines from horizontal axis
    horizontal_axis = original_image[y2:y2 + 10, x1:x2]
    find_grid_lines(original_image, image, dimensions, horizontal_axis)

    # Rotate image and repeat with other axis
    original_rotated_image = cv2.rotate(original_image, cv2.ROTATE_90_COUNTERCLOCKWISE)
    rotated_image = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)

    original_rotated_dimensions = rotate_dimension(x1, y1, x2, y2, original_image.shape[1], original_image.shape[0])
    original_rotated_x1, original_rotated_y1, original_rotated_x2, original_rotated_y2 = original_rotated_dimensions

    vertical_axis = original_rotated_image[original_rotated_y2:original_rotated_y2 + 10,
                    original_rotated_x1:original_rotated_x2]

    find_grid_lines(original_rotated_image, rotated_image, original_rotated_dimensions, vertical_axis)
    image = cv2.rotate(rotated_image, cv2.ROTATE_90_CLOCKWISE)

    return image


def to_colour(pixel):
    """
    Method to class all summed RGB values < 500 as Black, and the rest as white. Solely used to find axis ticks for the gridlines
    """
    if sum(pixel) < 500:
        return "BLACK"
    else:
        return "WHITE"


def find_grid_lines(original_image, image, dimensions, axis):
    """
    Method to find axis ticks. We assume all gridlines stem from axis ricks
    """
    human_readable = []

    for i, row in enumerate(axis):
        human_readable.append(list(map(to_colour, row)))

    consecutive_count = 0
    for i, row in enumerate(human_readable):

        # A column with axis ticks should only have 2 colors. White for the background, black for ticks.
        if len(set(row)) == 2:
            consecutive_count += 1
            # Ensure it wasn't coincidence
            if consecutive_count == 3:
                row_with_ticks = i
                break
        else:
            consecutive_count = 0  # Reset the counter if the condition is not met

    indices = [index for index, element in enumerate(human_readable[row_with_ticks]) if element == "BLACK"]

    color = (255, 255, 255)
    thickness = 2  # Assumed grid line length
    height = original_image.shape[0]

    for x in indices:
        start_point = (x, 0)
        end_point = (x, height)
        cv2.line(image[dimensions[1]:dimensions[3], dimensions[0]:dimensions[2]], start_point, end_point, color,
                 thickness)
    return image


def rotate_dimension(x1, y1, x2, y2, image_width, image_height):
    """
    Rotate dimension 90 degrees counter-clockwise
    """

    new_x1 = y1
    new_y1 = image_width - x2
    new_x2 = y2
    new_y2 = image_width - x1

    return new_x1, new_y1, new_x2, new_y2


def ocr(image_path):
    """
    Text detection with Google Vision API
    """

    def analyze_image_from_local_path(image_path: str, feature_types: Sequence) -> vision.AnnotateImageResponse:
        """
        Google Vision API text detection request
        """
        client = vision.ImageAnnotatorClient()
        with open(image_path, 'rb') as image_file:
            image_content = image_file.read()

        image = vision.Image(content=image_content)
        features = [vision.Feature(type_=feature_type) for feature_type in feature_types]
        request = vision.AnnotateImageRequest(image=image, features=features)

        response = client.annotate_image(request=request)
        return response

    def filter_and_create_dict(response: vision.AnnotateImageResponse):
        """
        Create dict with bounding boxes as keys, text as values
        """
        text_dict = {}
        for annotation in response.text_annotations[1:]:  # Skip the first element
            text = annotation.description
            vertices = annotation.bounding_poly.vertices
            bounding_box = ((vertices[0].x, vertices[0].y), (vertices[2].x, vertices[2].y))
            text_dict[bounding_box] = text
        return text_dict

    features = [vision.Feature.Type.TEXT_DETECTION]
    response = analyze_image_from_local_path(image_path, features)
    number_dict = filter_and_create_dict(response)

    # Load the image with OpenCV
    image = cv2.imread(image_path)

    # Draw bounding boxes on the image
    for ((start_x, start_y), (end_x, end_y)), _ in number_dict.items():
        # Define the rectangle color and thickness
        color = (0, 255, 0)  # Green
        thickness = 2
        # Draw the rectangle
        cv2.rectangle(image, (start_x, start_y), (end_x, end_y), color, thickness)

    # Save the image with bounding boxes
    cv2.imwrite("test.png", image)

    return number_dict


def preprocess_dict(text_dict):
    """
    Using regex to preprocess the numbers detected by Google Vision API
    """
    processed_dict = {}
    pattern = r'-?\d+\.\d+'

    for bounding_box, text in text_dict.items():
        numbers = re.findall(pattern, text)
        processed_dict[bounding_box] = ' '.join(numbers)

    return processed_dict


def adjust_offset(numbers_dict, offset):
    """
    Adjust for dimension offset of detecting numbers in a cropped image to reflect them across the entire image
    """
    adjusted_dict = {}
    offset_x, offset_y = offset

    for ((x1, y1), (x2, y2)), text in numbers_dict.items():
        # Adjust coordinates
        adjusted_x1 = x1 + offset_x
        adjusted_y1 = y1 + offset_y
        adjusted_x2 = x2 + offset_x
        adjusted_y2 = y2 + offset_y

        # Update in new dictionary
        adjusted_dict[((adjusted_x1, adjusted_y1), (adjusted_x2, adjusted_y2))] = text

    return adjusted_dict


def remove_numbers(numbers, plot):
    """
    Remove pre-existing colors by coloring bounding box columns by nearest neighbour
    """
    for (top_left, bottom_right), text in numbers.items():
        # Adjust coordinates to include one extra pixel on each side & Ensure we don't go out of bounds
        top_left_expanded = (max(int(top_left[0]) - 2, 0), max(int(top_left[1]) - 2, 0))
        bottom_right_expanded = (
            min(int(bottom_right[0]) + 2, plot.shape[1]), min(int(bottom_right[1]) + 2, plot.shape[0]))

        # Iterate through each column in the expanded bounding box
        for x in range(top_left_expanded[0], bottom_right_expanded[0]):
            # Get the pixels directly above the top of the expanded bounding box
            if top_left_expanded[1] - 3 >= 0:  # Check if there are 3 pixels above
                above_pixels = plot[top_left_expanded[1] - 3:top_left_expanded[1], x]
                # Calculate the mean color of 3 pixels
                mean_color = np.mean(above_pixels, axis=0)
                # Color the entire column by mean color
                plot[top_left_expanded[1]:bottom_right_expanded[1], x] = mean_color

    cv2.imwrite("test2.png", plot)
    return plot


def is_color_dark(color):
    """
    Calculate luminance of the color
    """

    luminance = (0.299 * color[2] + 0.587 * color[1] + 0.114 * color[0])
    return luminance < 128  # Threshold for dark /light


def draw_text_in_boxes(plot, numbers, font_scale=1.2, outline_offset=0.5):
    """
    Draw Google Vision API detected text with custom font
    """
    pil_image = Image.fromarray(cv2.cvtColor(plot, cv2.COLOR_BGR2RGB))
    font_path = "coolvetica rg.otf"

    for (top_left, bottom_right), text in numbers.items():
        box_width = bottom_right[0] - top_left[0]
        box_height = bottom_right[1] - top_left[1]

        font_size = int(font_scale * box_height)
        font = ImageFont.truetype(font_path, font_size)
        draw = ImageDraw.Draw(pil_image)

        text_parts = text.split()
        text_width_total = sum([draw.textlength(part, font=font) for part in text_parts])
        spacing = (box_width - text_width_total) / (len(text_parts) + 1)

        x_position = top_left[0] + spacing
        y_position = bottom_right[1] - font_size

        for index, part in enumerate(text_parts):
            check_x = top_left[0] if index == 0 else bottom_right[0] if index == len(text_parts) - 1 else (top_left[0] +
                                                                                                           bottom_right[
                                                                                                               0]) // 2
            check_y = top_left[1] - 1  # If pixel above is dark, we assume the number has a dark background, and we
            # color it white (and vice versa)
            pixel_color = plot[check_y, check_x]
            text_color = (255, 255, 255) if is_color_dark(pixel_color) else (0, 0, 0)
            outline_color = (0, 0, 0) if text_color == (255, 255, 255) else (255, 255, 255)

            # Draw outline to make it more distinguishable
            for dx in [-outline_offset, 0, outline_offset]:
                for dy in [-outline_offset, 0, outline_offset]:
                    if dx != 0 or dy != 0:
                        draw.text((x_position + dx, y_position + dy), part, font=font, fill=outline_color)

            draw.text((x_position, y_position), part, font=font, fill=text_color)

            text_width = draw.textlength(part, font=font)
            x_position += text_width + spacing

    image_with_text = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
    return image_with_text


def are_all_numbers(text):
    """
    Method to check if all detected elements are valid numbers & no elements are empty
    """
    pattern = r'^-?\d+(\.\d+)?$'

    if not text:  # Check if the string is empty
        return False
    return all(re.match(pattern, part) for part in text.split())


def detect_grid(input_image_path: str):
    """
    Using Yolov8 to detect the presence of a grid in a heatmap
    """
    model = YOLO("Object Detection Models/detect_grid.pt")
    grid = False
    results = model(input_image_path)
    confidences = results[0].boxes.conf

    if len(confidences) < 1:
        return grid

    max_conf_index = confidences.argmax().item()

    if confidences[max_conf_index] > 0.8:
        grid = True
        return grid

    return grid
