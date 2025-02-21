clc;
clear;
close all;
imtool close all;

%% Define the dataset path and initialize variables
base_path = "dataset/split/test";
image_files = dir(fullfile(base_path, "*.jpg"));
nfiles = length(image_files);
accuracies = zeros(1, nfiles);
metrics = cell(nfiles, 5); % Columns: IoU, TPR, FPR, Precision, Image name

totalFiles = nfiles; % Track total valid files

%% Initialize timers for performance tracking
start_time = tic;
last_500_time = tic;

%% Loop through each image in the dataset
for j = 1:nfiles
    image_filename = image_files(j).name;
    label_filename = strrep(image_filename, ".jpg", ".png");
    
    filename = fullfile(base_path, image_filename);
    filename_label = fullfile(base_path, label_filename);
    
    % Skip if corresponding label file does not exist
    if ~isfile(filename_label)
        metrics(j, :) = {0, 0, 0, 0, image_filename};
        totalFiles = totalFiles - 1;
        continue;
    end
    
    % Read image and label
    img = imread(filename);
    img_label = imbinarize(imread(filename_label));
    
    % Convert image to HSV color space
    hsv_img = rgb2hsv(img);
    h = hsv_img(:,:,1);
    s = hsv_img(:,:,2);
    v = hsv_img(:,:,3);
    
    % Define lane detection masks based on HSV thresholds
    lane_mask_interior = (h > 0.7 & s < 0.04 & s > 0.02 & v > 0.70 & v < 0.79);
    lane_mask_exterior = ((h > 0.06 & h < 0.12) & (s > 0.02 & s < 0.15) & v > 0.83);
    
    % Apply morphological operations for noise reduction
    diam2 = strel('diamond', 2);
    processed_interior = imclose(lane_mask_interior, diam2);
    processed_exterior = imclose(lane_mask_exterior, diam2);
    result = processed_interior + processed_exterior;
    
    % Compute evaluation metrics
    [IoU, TPR, FPR, Precision] = computeStats(result, img_label);
    metrics(j, :) = {IoU, TPR, FPR, Precision, image_filename};
    
    % Display progress every 500 images
    if mod(j, 500) == 0
        elapsed_500 = toc(last_500_time);
        fprintf("Time for last 500 images: %.2f seconds\n", elapsed_500);
        last_500_time = tic;
    end
    
    % Estimate remaining time every 1000 images
    if mod(j, 1000) == 0 || j == nfiles
        fprintf("Processed %d/%d images. Estimated time remaining: %.2f seconds.\n", j, nfiles, toc(start_time) / j * (nfiles - j));
    end
end

%% Handle NaN values in metrics
for i = 1:4
    metrics(isnan(cell2mat(metrics(:, i))), i) = {0};
end

%% Compute average metrics over valid files
iou = sum(cell2mat(metrics(:, 1))) / totalFiles;
tpr = sum(cell2mat(metrics(:, 2))) / totalFiles;
fpr = sum(cell2mat(metrics(:, 3))) / totalFiles;
precision = sum(cell2mat(metrics(:, 4))) / totalFiles;

%% Display final performance metrics
fprintf("IoU: %.2f%%\n", iou);
fprintf("TPR: %.2f%%\n", tpr);
fprintf("FPR: %.2f%%\n", fpr);
fprintf("Precision: %.2f%%\n", precision);

%% Function to compute evaluation metrics
function [IoU, TPR, FPR, Precision] = computeStats(testMask, truthMask)
    if ~isequal(size(testMask), size(truthMask))
        error('Input images must have the same dimensions');
    end
    
    intersection = testMask & truthMask;
    union = testMask | truthMask;
    IoU = sum(intersection(:)) / sum(union(:));
    
    truePositives = sum(intersection(:));
    actualPositives = sum(truthMask(:));
    TPR = truePositives / actualPositives;
    
    falsePositives = sum(testMask(:)) - truePositives;
    actualNegatives = sum(~truthMask(:));
    FPR = falsePositives / actualNegatives;
    
    predictedPositives = sum(testMask(:));
    Precision = truePositives / predictedPositives;
end

%% Function to visualize an image, its ground truth, and prediction
function displayAndAnalyzeImage(filename, base_path)
    image_filename = fullfile(base_path, filename);
    label_filename = strrep(image_filename, ".jpg", ".png");
    
    % Check if label file exists
    if ~isfile(label_filename)
        error("Label file does not exist for %s", filename);
    end
    
    % Read and preprocess image
    img = imread(image_filename);
    img_label = imbinarize(imread(label_filename));
    
    % Convert to HSV and apply lane detection filters
    hsv_img = rgb2hsv(img);
    h = hsv_img(:,:,1);
    s = hsv_img(:,:,2);
    v = hsv_img(:,:,3);
    
    lane_mask_interior = h > 0.7 & s < 0.04 & s > 0.02 & v > 0.70 & v < 0.79;
    lane_mask_exterior = (h > 0.06 & h < 0.12) & (s > 0.02 & s < 0.15) & v > 0.83;
    
    diam2 = strel('diamond', 2);
    processed_interior = imclose(lane_mask_interior, diam2);
    processed_exterior = imclose(lane_mask_exterior, diam2);
    result = processed_interior + processed_exterior;
    
    % Compute metrics
    [IoU, TPR, FPR, Precision] = computeStats(result, img_label);
    
    % Display results
    figure;
    subplot(1, 3, 1); imshow(img); title('Original Image');
    subplot(1, 3, 2); imshow(img_label); title('Ground Truth');
    subplot(1, 3, 3); imshow(result); title('Predicted Mask');
    
    % Print metrics
    fprintf("IoU: %.4f\n", IoU);
    fprintf("TPR: %.4f\n", TPR);
    fprintf("FPR: %.4f\n", FPR);
    fprintf("Precision: %.4f\n", Precision);
end

%% Example function calls for visualization
displayAndAnalyzeImage("000_Both_left_curve_0009.jpg", "dataset/split/test");
