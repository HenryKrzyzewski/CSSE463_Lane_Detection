clc;
clear;
close all;
imtool close all;

base_path = "dataset/FixData";
image_files = dir(fullfile(base_path, "*.jpg"));
nfiles = length(image_files);

start_time = tic;
last_500_time = tic;

total_IOU = 0;
highest_IOU = 0;
lowest_IOU = 1;
highest_IOU_filename = "";
lowest_IOU_filename = "";

total_TP = 0;
total_FP = 0;
total_TN = 0;
total_FN = 0;
for j = 700:700
    image_filename = image_files(j).name;
    label_filename = strrep(image_filename, ".jpg", ".png");
    
    filename = fullfile(base_path, image_filename);
    filename_label = fullfile(base_path, label_filename);
    
    if ~isfile(filename_label)
        continue; % Skip if label file doesn't exist
    end

    img_truth = imread(filename_label);
    temp = imbinarize(img_truth);

    img = imread(filename);
    [height, width, num_channels] = size(img);
    img_label = imbinarize(imread(filename_label));

    original_size = [height, width];

    model = load('net_checkpoint__1592__2025_02_17__13_28_21.mat');
    model = model.net;
    image_size = [256 256];
    img = imresize(img, image_size);
    prediction = semanticseg(img, model);
    pred_mask = zeros(size(prediction));
    pred_mask(prediction == 'C1') = 1;

    pred_mask = imresize(pred_mask, original_size);
    
    pred_mask(1:floor(end/2.7), :) = 0;

    imtool(pred_mask);

    filtered_result = bwareaopen(pred_mask, 300);

    se_disk = strel('disk', 5);
    se_diam = strel('diamond', 5);
    filtered_result = imclose(filtered_result, se_disk);

    imtool(filtered_result);

    cc = bwconncomp(filtered_result);
    stats = regionprops(cc, 'PixelList');

    img = imresize(img, original_size);
    % figure; imshow(img); hold on;
    title(['Lane Detection - Image ', num2str(j)]);

    % Create a blank bitmask for the lines
    line_mask = zeros(height, width);

    y_min_threshold = height * 0.31;
    y_max_threshold = height * 0.8;

    for i = 1:numel(stats)
        pixels = stats(i).PixelList;
        if size(pixels,1) < 2
           continue;
        end
    
        region_mask = false(height, width);
        idx = sub2ind([height, width], pixels(:,2), pixels(:,1));
        region_mask(idx) = true;

        imtool(region_mask)
        
        [H, theta, rho] = hough(region_mask);
        
        peaks = houghpeaks(H, 1, 'Threshold', ceil(0.3 * max(H(:))));
        
        lines = houghlines(region_mask, theta, rho, peaks, 'FillGap', 5, 'MinLength', 7);
        
        if isempty(lines)
             continue;
        end
        
        line_lengths = arrayfun(@(l) norm(l.point1 - l.point2), lines);
        [~, longest_idx] = max(line_lengths);
        selected_line = lines(longest_idx);
        
        pt1 = selected_line.point1;
        pt2 = selected_line.point2;
        
        if pt2(1) ~= pt1(1)
            slope = (pt2(2) - pt1(2)) / (pt2(1) - pt1(1));
            intercept = pt1(2) - slope * pt1(1);
        else
            continue;
        end
        
        y1_extended = height;    
        y2_extended = height * 0.3;  
        
        x1_extended = (y1_extended - intercept) / slope;
        x2_extended = (y2_extended - intercept) / slope;
        
        plot([x1_extended, x2_extended], [y1_extended, y2_extended], 'r-', 'LineWidth', 6);
        
        line_mask = insertShape(line_mask, 'Line', [x1_extended, y1_extended, x2_extended, y2_extended], 'Color', 'white', 'LineWidth', 20);
    end

    line_mask = im2bw(line_mask);

    imtool(img);
    imtool(line_mask);

    hold off;

    intersection = sum((line_mask(:) & img_label(:))); 
    union = sum((line_mask(:) | img_label(:)));  

    if union == 0
        IoU = 0;
    else
        IoU = intersection / union;
    end

    total_IOU = total_IOU + IoU;

    if IoU > highest_IOU
        highest_IOU = IoU;
        highest_IOU_filename = image_filename;
    end

    if IoU < lowest_IOU
        lowest_IOU = IoU;
        lowest_IOU_filename = image_filename;
    end

    TP = sum(line_mask(:) & img_label(:));
    FP = sum(line_mask(:) & ~img_label(:));
    TN = sum(~line_mask(:) & ~img_label(:));
    FN = sum(~line_mask(:) & img_label(:));

    total_TP = total_TP + TP;
    total_FP = total_FP + FP;
    total_TN = total_TN + TN;
    total_FN = total_FN + FN;

    precision = TP / (TP + FP);
    TPR = TP / (TP + FN);
    FPR = FP / (FP + TN);


    fprintf("IoU for image %d (%s): %.4f\nTP: %.4f\nFP: %.4f\nPrecision: %.4f\n\n", ...
            j, image_filename, IoU, TPR, FPR, precision);
  

    % Display processed binary mask
    % figure; imshow(filtered_result);
    % title('Processed Mask');

    if mod(j, 500) == 0
        elapsed_500 = toc(last_500_time);
        fprintf("Time for last 500 images: %.2f seconds\n", elapsed_500);
        last_500_time = tic;
    end

    if mod(j, 1000) == 0 || j == nfiles
        fprintf("Processed %d/%d images. Estimated time remaining: %.2f seconds.\n", ...
                j, nfiles, toc(start_time) / j * (nfiles - j));
    end
end

%% new

mean_IOU = total_IOU / 1264;
TPR = total_TP / max(total_TP + total_FN, 1); % Sensitivity
FPR = total_FP / max(total_FP + total_TN, 1); % False Positive Rate
TNR = total_TN / max(total_TN + total_FP, 1); % Specificity
FNR = total_FN / max(total_FN + total_TP, 1); % Miss Rate

fprintf("\n===== Overall Results =====\n");
fprintf("Mean IoU: %.4f\n", mean_IOU);
fprintf("TPR (Recall): %.4f\n", TPR);
fprintf("FPR (Fall-out): %.4f\n", FPR);
fprintf("TNR (Specificity): %.4f\n", TNR);
fprintf("FNR (Miss Rate): %.4f\n", FNR);
fprintf("Highest IoU: %.4f (%s)\n", highest_IOU, highest_IOU_filename);
fprintf("Lowest IoU: %.4f (%s)\n", lowest_IOU, lowest_IOU_filename);

imwrite(imread(fullfile(base_path, highest_IOU_filename)), "highest_IOU_image.jpg");
imwrite(imread(fullfile(base_path, lowest_IOU_filename)), "lowest_IOU_image.jpg");

fprintf("Saved highest and lowest IoU images.\n");
