clc
clear
close all
imtool close all
training_path = "dataset\CarlaData\train\";
training_label_path = "dataset\CarlaData\train_label";
dinfo = dir(fullfile(training_path));
dinfo_label = dir(fullfile(training_label_path));
dinfo([dinfo.isdir]) = [];
dinfo_label([dinfo_label.isdir]) = [];

%Sort filenames naturally
train_filenames = sort_nat({dinfo.name});
label_filenames = sort_nat({dinfo_label.name});

nfiles = length(dinfo);
accuracies = zeros(1, nfiles); % Preallocate for efficiency
for j = 1 : nfiles
%for j = 1:100
    filename = fullfile(training_path, train_filenames{j});
    filename_label = fullfile(training_label_path, label_filenames{j});

    img = imread(filename);
    img_label = imbinarize(imread(filename_label));
    %imtool(img)
    hsv_img = rgb2hsv(img);
    h = hsv_img(:,:,1);
    s = hsv_img(:,:,2);
    v = hsv_img(:,:,3);
    %imtool(hsv_img)
    lane_mask_interior = h > 0.67 & s < 0.02 & v > 0.8;
    lane_mask_exterior = (h > 0.48 & h < 0.55) & (s > 0.04 & s < 0.12) & v > 0.83;

    diam2 = strel('diamond', 2);
    processed_interior = imclose(lane_mask_interior, diam2);
    %imtool(processed_interior)
    processed_exterior = imclose(lane_mask_exterior, diam2);
    %imtool(processed_exterior)
    result = processed_interior + processed_exterior;
    %imtool(result)
    %imtool(img_label)
    intersection = result & img_label;
    union = result | img_label;
    IoU = sum(intersection(:)) / sum(union(:));

    accuracies(j) = IoU; % Store IoU in the array
end

accuracy = (sum(accuracies) / length(accuracies))*100

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


