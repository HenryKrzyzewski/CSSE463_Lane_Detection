clc
clear
close all
imtool close all

% Define the dataset path
base_path = "dataset\split\test";

% Get all JPG files
image_files = dir(fullfile(base_path, "*.jpg"));

nfiles = length(image_files);
nfiles = 1000;
accuracies = zeros(1, nfiles); % Preallocate for efficiency

% Initialize counters and timers
start_time = tic;
last_500_time = tic;

%for j = 1 : nfiles
totalFiles = nfiles;
for j = 1:nfiles
    image_filename = image_files(j).name;
    label_filename = strrep(image_filename, ".jpg", ".png");
    
    filename = fullfile(base_path, image_filename);
    filename_label = fullfile(base_path, label_filename);
    
    if ~isfile(filename_label)
        IoU = 0;
        accuracies(j) = IoU;
        totalFiles = totalFiles - 1;
        continue; % Skip if matching label file doesn't exist
    end

    img = imread(filename);
    img_label = imbinarize(imread(filename_label));
    %imtool(img)
    hsv_img = rgb2hsv(img);
    h = hsv_img(:,:,1);
    s = hsv_img(:,:,2);
    v = hsv_img(:,:,3);
    %imtool(hsv_img)
    lane_mask_interior = h > 0.7 & s < 0.04 & s > 0.02 & v > 0.70 & v < 0.79;
    lane_mask_exterior = (h > 0.06 & h < 0.12) & (s > 0.02 & s < 0.15) & v > 0.83;
    %lane_mask_exterior = 0;
    diam2 = strel('diamond', 2);
    processed_interior = imclose(lane_mask_interior, diam2);
    %imtool(processed_interior)
    processed_exterior = imclose(lane_mask_exterior, diam2);
    %imtool(processed_exterior)
    result = processed_interior + processed_exterior;
    %imtool(result)
    %imtool(img_label)

    [IoU, TPR, FPR, Precision] = computeStats(result, img_label);
    accuracies(j) = IoU; % Store IoU in the array
    % Timing for every 500 images
    if mod(j, 500) == 0
        elapsed_500 = toc(last_500_time);
        fprintf("Time for last 500 images: %.2f seconds\n", elapsed_500);
        last_500_time = tic; % Reset the timer
    end

    % Progress update
    if mod(j, 1000) == 0 || j == nfiles
        fprintf("Processed %d/%d images. Estimated time remaining: %.2f seconds.\n", j, nfiles, toc(start_time) / j * (nfiles - j));
    end
end
%length(accuracies)
accuracy = (sum(accuracies) / totalFiles)*100



function [IoU, TPR, FPR, Precision] = computeStats(testMask, truthMask)
    % computeStats calculates the Intersection over Union (IoU), True Positive Rate (TPR),
    % False Positive Rate (FPR), and Precision between two binary images.
    % Inputs:
    %   testMask - First binary image (logical matrix)
    %   truthMask - Second binary image (logical matrix)
    % Outputs:
    %   IoU - Intersection over Union value
    %   TPR - True Positive Rate
    %   FPR - False Positive Rate
    %   Precision - Positive Predictive Value

    if ~isequal(size(testMask), size(truthMask))
        error('Input images must have the same dimensions');
    end
    
    % Compute intersection and union
    intersection = testMask & truthMask;
    union = testMask | truthMask;
    
    % Compute IoU
    IoU = sum(intersection(:)) / sum(union(:));
    
    % Compute True Positive Rate (TPR)
    truePositives = sum(intersection(:));
    actualPositives = sum(truthMask(:));
    TPR = truePositives / actualPositives;
    
    % Compute False Positive Rate (FPR)
    falsePositives = sum(testMask(:)) - truePositives;
    actualNegatives = sum(~truthMask(:));
    FPR = falsePositives / actualNegatives;
    
    % Compute Precision
    predictedPositives = sum(testMask(:));
    Precision = truePositives / predictedPositives;
end



function sorted_files = sort_nat(file_list)
    % Extract numeric parts from filenames
    expr = '(\d+)'; % Regex to find numbers
    num_list = zeros(size(file_list));
    
    for i = 1:length(file_list)
        tokens = regexp(file_list{i}, expr, 'match');
        if ~isempty(tokens)
            num_list(i) = str2double(tokens{end}); % Use last number in filename
        end
    end
    
    % Sort based on extracted numbers
    [~, sorted_idx] = sort(num_list);
    sorted_files = file_list(sorted_idx);
end


