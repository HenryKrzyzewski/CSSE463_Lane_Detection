clc;
clear;
close all;
imtool close all;

% Define dataset path
base_path = "dataset/FixData";
image_files = dir(fullfile(base_path, "*.jpg"));
nfiles = length(image_files);

% Initialize timing
start_time = tic;
last_500_time = tic;

for j = 300:300
    % Load image and corresponding label
    image_filename = image_files(j).name;
    label_filename = strrep(image_filename, ".jpg", ".png");
    
    filename = fullfile(base_path, image_filename);
    filename_label = fullfile(base_path, label_filename);
    
    if ~isfile(filename_label)
        continue; % Skip if label file doesn't exist
    end

    img = imread(filename);
    img = imresize(img, 1.0, 'nearest');
    image_width = size(img, 2); % Get full image width
    img_label = imbinarize(imread(filename_label));
    
    % Convert to HSV and extract channels
    hsv_img = rgb2hsv(img);
    h = hsv_img(:,:,1);
    s = hsv_img(:,:,2);
    v = hsv_img(:,:,3);
    
    % Lane detection masks
    lane_mask_white1 = (h > 0.05 & h < 0.12) & (s < 0.12 & s > 0.05) & (v > 0.82 & v < 0.95);
    lane_mask_white2 = (h > 0.64 & h < 0.8) & (s < 0.05 & s > 0.00) & (v > 0.7 & v < 0.87);
    lane_mask = lane_mask_white1 | lane_mask_white2;

    % Morphological operations
    se_disk = strel('disk', 6);
    se_diam = strel('diamond', 3);
    se_vert_line = strel('line', 5, 90);
    se_vert_line_right = strel('line', 10, 60);
    
    filtered_result = bwareaopen(lane_mask, 80);
    processed_mask = imclose(filtered_result, se_vert_line);
    % processed_mask = imdilate(processed_mask, se_diam);
    % processed_mask = imdilate(processed_mask, se_vert_line);
    % processed_mask = imerode(processed_mask, se_disk);

    % Identify connected components (individual lane markings)
    cc = bwconncomp(processed_mask);
    stats = regionprops(cc, 'PixelList');  % Extract pixel coordinates

    % Display original image
    figure; imshow(img); hold on;
    title(['Lane Detection - Image ', num2str(j)]);

    [img_height, img_width, ~] = size(img);
    y_min_threshold = img_height * 0.31;
    y_max_threshold = img_height * 0.8;

    % Process each region separately
    for i = 1:numel(stats)
        % Extract pixel coordinates for this region
        pixels = stats(i).PixelList;
        if isempty(pixels)
            continue; % Skip empty regions
        end

        % Compute bounding box manually
        xMin = min(pixels(:,1));
        xMax = max(pixels(:,1));
        yMin = min(pixels(:,2));
        yMax = max(pixels(:,2));

        if yMax < y_min_threshold || yMin > y_max_threshold
            continue;
        end

        % Ensure valid cropping dimensions
        width = max(1, xMax - xMin + 1);
        height = max(1, yMax - yMin + 1);

        % Create a cropped mask for the current region
        region_mask = false(size(processed_mask));
        idx = sub2ind(size(region_mask), pixels(:,2), pixels(:,1)); % Convert pixel coordinates to index
        region_mask(idx) = true;
        region_crop = imcrop(region_mask, [xMin, yMin, width, height]);

        % Apply Hough Transform to the cropped region
        [H, theta, rho] = hough(region_crop);
        peaks = houghpeaks(H, 1, 'threshold', ceil(0.3 * max(H(:)))); % Get strongest line
        lines = houghlines(region_crop, theta, rho, peaks, 'FillGap', 10, 'MinLength', 30);

        if isempty(lines)
            continue; % Skip if no lines detected
        end

        % Select the strongest line (longest detected)
        max_len = 0;
        best_line = [];
        for k = 1:length(lines)
            p1 = lines(k).point1;
            p2 = lines(k).point2;
            line_len = norm(p1 - p2);
            if line_len > max_len
                max_len = line_len;
                best_line = lines(k);
            end
        end

        % Ensure a line was selected
        if ~isempty(best_line)
            p1 = best_line.point1;
            p2 = best_line.point2;

            % Compute slope and intercept of the selected line
            slope = (p2(2) - p1(2)) / (p2(1) - p1(1) + eps); % Avoid division by zero
            intercept = p1(2) - slope * p1(1);

            % Compute the center of the detected white dash
            xCenter = mean(pixels(:,1));
            yCenter = mean(pixels(:,2));

            % Recompute the intercept to make the line pass through the center
            intercept = yCenter - slope * xCenter;

            % Extend the line symmetrically across the full image width
            extension_length = max(xMax - xMin, yMax - yMin) * 2; % Extend beyond region
            x1_extended = xCenter - extension_length / 2;
            x2_extended = xCenter + extension_length / 2;
            y1_extended = slope * x1_extended + intercept;
            y2_extended = slope * x2_extended + intercept;

            % Plot corrected lane line centered on the white dash
            plot([x1_extended, x2_extended], [y1_extended, y2_extended], 'r-', 'LineWidth', 3);
        end
    end

    hold off;

    % Display processed binary mask
    figure; imshow(processed_mask);
    title('Processed Mask');

    % Timing for every 500 images
    if mod(j, 500) == 0
        elapsed_500 = toc(last_500_time);
        fprintf("Time for last 500 images: %.2f seconds\n", elapsed_500);
        last_500_time = tic;
    end

    % Progress update
    if mod(j, 1000) == 0 || j == nfiles
        fprintf("Processed %d/%d images. Estimated time remaining: %.2f seconds.\n", ...
                j, nfiles, toc(start_time) / j * (nfiles - j));
    end
end
