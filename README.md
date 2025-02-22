# Setup (Dataset Structure)
Download the dataset and place it in the dataset folder (a .gitkeep file should keep it in the repo; the dataset itself is gitignore'd). The file structure should go dataset > split > (train/test/validation) > (individual files)
Original: https://glutamat42.github.io/Ultra-Fast-Lane-Detection/
Modified and cleaned dataset: https://rosehulman-my.sharepoint.com/:u:/g/personal/gaulldj_rose-hulman_edu/EU38xbctdNFHrWE2X5o7a8UB22JSxafzbOccgw3YSnNNQw?e=fhrWMT

# Required MATLAB Toolboxes:
* Image Processing Toolbox
* Deep Learning Toolbox

# Running
* Baseline.m: This script processes a test dataset of images to evaluate lane detection performance using HSV-based color thresholding. The images are segmented into lane and non-lane areas, followed by morphological processing to improve detection accuracy. The script computes evaluation metrics such as Intersection over Union (IoU), True Positive Rate (TPR), False Positive Rate (FPR), and Precision.
* Minimum.m: trains an AlexNet-based CNN model to detect lanes in images. The process involves splitting images into patches, categorizing them, resizing them for AlexNet, and training the network. The final model is evaluated on test images using metrics such as IoU, TPR, FPR, and Precision.
* computeMaskMetrics.m: Function used to calculates IoU, TPR, FPR, and Precision to evaluate the similarity between two binary masks.
* run_deeplab.m: This file will run our final DeepLab model on any image of choice. You can provide a file path near the top and it will display the image and corresponding predicted lane mask.
* stretch.m: This file is still in progress and you can run it on any subset of our data available in our dataset.
