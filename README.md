## Recoloring Problematic Images For Colorblindness

| # | Filename | Function |
|----------|----------|----------|
| 0 | main.py | Takes in image path or directory of images from the user |
| 1 | color_extraction.py | Uses Yolov8 to detect and isolate legend and it's distribution (i.e. qualitative, diverging, sequential), the heatmap dimensions (if present), and extracts color form the legend |
| 2| colormap_selection.py | Following color extraction, we simulate colorblindness and create color distance matrices. We evaluate the to select the best colormap (NOTE: the evaluation was only used at the start to determine the best color pallete for each color vision deficiency for each distribution) |
| 3 | apply_colormap.py | Colormapping from original to new color pallete is created and applied, with further refinements such as interpolation |
| 4 | refinements.py | For heatmaps, the Yolov8 detected heatmap dimensions are refined. For discrete plots, soft gauissian blur is applied |
| 5 | gridlines.py | Yolov8 is used to detect the presence of gridlines. If present they are redrawn here. Google Vision API is used to detect the presence of numbers (OCR), if present, they are redrawn. |
| 6 | process_image.py| In charge of calling and coordinating methods in the other files (excluding main.py, which calls process_image.py). Validation also takes place here, where it is determined if the image needs recoloring.   |
| 7 | [Folder] Object Detection Models | Contains Yolov8 detection models: 'best.pt' detects legends and legend distributions; 'detect_grid.pt' detects the presence of gridlines; 'heatmap_detect.pt' detects heatmap dimensions|
| 8 | coolvetica_rg.otf | Font used in redrawing numbers detected by Google Vision API |


## Documentation

Information regarding libraries, dependencies, method descriptions and methodologies is detailed in thesis.

