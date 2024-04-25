import numpy as np
import cv2
import colorama
from colorama import Fore, Back, Style


def adjust_bounding_box(image, bbox, threshold=0.5, black_threshold=2, white_threshold=255):
    """
    Adjust bounding box to be within black axis
    """
    x, y, x2, y2 = bbox
    height, width = image.shape[:2]

    def is_mostly_black_or_white(line, threshold, black_threshold, white_threshold):
        """
         Check if a line is made up of mostly black/white pixels
        """
        white_pixels = np.sum(np.all(line >= white_threshold, axis=-1))
        black_pixels = np.sum(np.all(line <= black_threshold, axis=-1))

        return (white_pixels + black_pixels) / len(line) > threshold

    #Keep shrinking dimensions until each one lies on a mostly black line (when there's an axis) or mostly white when no axis is present
    while True:
        top_changed = bottom_changed = left_changed = right_changed = False

        # Check top edge and one row inward
        if y + 1 < height:
            top_edge = image[y, x:x2]
            next_row = image[y + 1, x:x2]
            if is_mostly_black_or_white(top_edge, threshold, black_threshold, white_threshold) or \
                    is_mostly_black_or_white(next_row, threshold, black_threshold, white_threshold):
                y += 1
                top_changed = True

        # Check bottom edge and one row inward
        if y2 - 1 > 0:
            bottom_edge = image[y2 - 1, x:x2]
            prev_row = image[y2 - 2, x:x2] if y2 - 2 >= 0 else bottom_edge
            if is_mostly_black_or_white(bottom_edge, threshold, black_threshold, white_threshold) or \
                    is_mostly_black_or_white(prev_row, threshold, black_threshold, white_threshold):
                y2 -= 1
                bottom_changed = True

        # Check left edge and one column inward
        if x + 1 < width:
            left_edge = image[y:y2, x]
            next_col = image[y:y2, x + 1]
            if is_mostly_black_or_white(left_edge, threshold, black_threshold, white_threshold) or \
                    is_mostly_black_or_white(next_col, threshold, black_threshold, white_threshold):
                x += 1
                left_changed = True

        # Check right edge and one column inward
        if x2 - 1 > 0:
            right_edge = image[y:y2, x2 - 1]
            prev_col = image[y:y2, x2 - 2] if x2 - 2 >= 0 else right_edge
            if is_mostly_black_or_white(right_edge, threshold, black_threshold, white_threshold) or \
                    is_mostly_black_or_white(prev_col, threshold, black_threshold, white_threshold):
                x2 -= 1
                right_changed = True

        # Stop if no edges changed
        if not (top_changed or bottom_changed or left_changed or right_changed):
            break

    # Expand dimensions (in case Yolov8 made a bounding box inside the plot, not around it)
    while True:
        expanded = False

        # Expand top edge
        if y > 0 and not is_mostly_black_or_white(image[y - 1, x:x2], threshold, black_threshold, white_threshold):
            y -= 1
            expanded = True

        # Expand bottom edge
        if y2 < height - 1 and not is_mostly_black_or_white(image[y2, x:x2], threshold, black_threshold,
                                                            white_threshold):
            y2 += 1
            expanded = True

        # Expand left edge
        if x > 0 and not is_mostly_black_or_white(image[y:y2, x - 1], threshold, black_threshold, white_threshold):
            x -= 1
            expanded = True

        # Expand right edge
        if x2 < width - 1 and not is_mostly_black_or_white(image[y:y2, x2], threshold, black_threshold,
                                                           white_threshold):
            x2 += 1
            expanded = True

        # Stop if no expansion occurred
        if not expanded:
            break

    return x, y, x2, y2


def apply_light_gaussian_blur(image):
    """
    Apply a soft gaussian blur across the image
    """

    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    black_threshold = 20
    white_threshold = 235

    mask = np.logical_and(gray_image > black_threshold, gray_image < white_threshold)
    mask = np.stack((mask, mask, mask), axis=-1)

    # Apply Gaussian blur to the entire image
    kernel_size = (3, 3)
    blurred_image = cv2.GaussianBlur(image, kernel_size, 0)

    result_image = np.where(mask, cv2.addWeighted(image, 0.6, blurred_image, 0.4, 0), image)

    return result_image


def get_deficiency_input():
    """
     Request input for user's CVD
    """
    colorama.init(autoreset=True)

    while True:
        deficiency = input(Fore.MAGENTA + Style.BRIGHT + Back.BLACK +
                           " What is your Colour Vision Deficiency? " + Back.RESET + "\n\n1. Protanopia\n2. Deuteranopia\n3. Tritanopia\n\n" + Fore.BLACK + Style.BRIGHT + Back.MAGENTA + "(Enter 1, 2, or 3):"
                           ).strip()

        if deficiency in ['1', '2', '3']:
            if deficiency == '1':
                deficiency = 'protan'
            elif deficiency == '2':
                deficiency = 'deutan'
            else:
                deficiency = 'tritan'

            return deficiency
        else:
            print(Fore.LIGHTYELLOW_EX + Style.BRIGHT + "Invalid input. Please enter 1, 2, or 3.")


def get_user_input():
    """
     Request input
     """
    while True:
        user_choice = input(
            "\n\n" + Fore.MAGENTA + Style.BRIGHT + Back.BLACK + " Would you like to process: " + Back.RESET + "\n\n1. A single image \n2. A batch of images\n\n" + Fore.BLACK + Style.BRIGHT + Back.MAGENTA + "(Enter 1 or 2): ").strip()

        if user_choice in ['1', '2']:
            return user_choice
        else:
            print(Fore.LIGHTYELLOW_EX + Style.BRIGHT + "Invalid input. Please enter 1 or 2.")
