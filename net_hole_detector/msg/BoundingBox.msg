string class_id      # "0", detected class label, e.g. "hole"
float64 score        # Detection confidence score, from 0.0 to 1.0

# Normalized bounding box coordinates, from 0.0 to 1.0.
# This follows the standard YOLO format and simplifies processing for the next node.
float64 x            # Bounding box center X
float64 y            # Bounding box center Y
float64 w            # Bounding box width
float64 h            # Bounding box height