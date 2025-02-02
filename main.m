training_path = "data/train";
training_label_path = "data/train_label";
dinfo = dir(fullfile(training_path));
dinfo_label = dir(fullfile(training_label_path));
dinfo([dinfo.isdir]) = [];
dinfo_label([dinfo_label.isdir]) = [];
nfiles = length(dinfo);
overall_accuracy = 0.0;
accuracies = zeros(1, nfiles); % Preallocate for efficiency
%for j = 1 : nfiles
%     filename = fullfile(training_path, dinfo(j).name);
%     filename_label = fullfile(training_label_path, dinfo_label(j).name);
% 
%     img = imread(filename);
%     img_label = imread(filename_label);
% 
%     hsv_img = rgb2hsv(img);
%     h = hsv_img(:,:,1);
%     s = hsv_img(:,:,2);
%     v = hsv_img(:,:,3);
% 
%     lane_mask_interior = h > 0.67 & s < 0.02 & v > 0.8;
%     lane_mask_exterior = (h > 0.48 & h < 0.55) & (s > 0.04 & s < 0.12) & v > 0.83;
% 
%     diam2 = strel('diamond', 2);
%     processed_interior = imclose(lane_mask_interior, diam2);
%     processed_exterior = imclose(lane_mask_exterior, diam2);
% 
%     result = processed_interior + processed_exterior;
% 
%     intersection = result & img_label;
%     union = result | img_label;
%     IoU = sum(intersection(:)) / sum(union(:));
% 
%     accuracies(j) = IoU; % Store IoU in the array
% 
%     overall_accuracy = overall_accuracy + IoU;
% end

% overall_accuracy = overall_accuracy / nfiles

filename = fullfile(training_path, "Town04_Clear_Noon_09_09_2020_14_57_22_frame_2.png");
filename_label = fullfile(training_label_path, "Town04_Clear_Noon_09_09_2020_14_57_22_frame_2_label.png");

img = imread(filename);
img_label = imread(filename_label);
img_label = imbinarize(img_label);

hsv_img = rgb2hsv(img);
h = hsv_img(:,:,1);
s = hsv_img(:,:,2);
v = hsv_img(:,:,3);

imtool(hsv_img);

lane_mask_interior = h > 0.69 & s < 0.02 & v > 0.85;
lane_mask_exterior = (h > 0.48 & h < 0.55) & (s > 0.04 & s < 0.12) & v > 0.83;

diam2 = strel('diamond', 2);
processed_interior = imclose(lane_mask_interior, diam2);
processed_exterior = imclose(lane_mask_exterior, diam2);

result = processed_interior + processed_exterior;

imtool(result)
imtool(img_label)

intersection = result & img_label;
imtool(intersection);
union = result | img_label;
imtool(union);
IoU = sum(intersection(:)) / sum(union(:))



