# Setup (Dataset Structure)
Download the dataset and place it in the dataset folder (a .gitkeep file should keep it in the repo; the dataset itself is gitignore'd). The file structure should go dataset > split > (train/test/validation) > (individual files)

# Required MATLAB Toolboxes:
* Image Processing Toolbox
* Deep Learning Toolbox

# Running
* Baseline: This script processes a test dataset of images to evaluate lane detection performance using HSV-based color thresholding. The images are segmented into lane and non-lane areas, followed by morphological processing to improve detection accuracy. The script computes evaluation metrics such as Intersection over Union (IoU), True Positive Rate (TPR), False Positive Rate (FPR), and Precision.
* Minimum: trains an AlexNet-based CNN model to detect lanes in images. The process involves splitting images into patches, categorizing them, resizing them for AlexNet, and training the network. The final model is evaluated on test images using metrics such as IoU, TPR, FPR, and Precision.

TODO:
