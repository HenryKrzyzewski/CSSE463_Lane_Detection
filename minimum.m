% function splitAndSortImages(imagePath, truthPath, outputDir)
%     % Read the input images
%     img = imread(imagePath);
%     truth = imread(truthPath);
% 
%     % Ensure the truth image is binary
%     truth = imbinarize(truth);
% 
%     % Get image dimensions
%     [height, width, ~] = size(img);
% 
%     % Define patch size
%     patchSize = 10;
% 
%     % Create output directories if they don't exist
%     partOfLaneDir = fullfile(outputDir, 'partOfLane');
%     notPartOfLaneDir = fullfile(outputDir, 'notPartOfLane');
%     if ~exist(partOfLaneDir, 'dir')
%         mkdir(partOfLaneDir);
%     end
%     if ~exist(notPartOfLaneDir, 'dir')
%         mkdir(notPartOfLaneDir);
%     end
% 
%     % Iterate over the image in 10x10 patches
%     patchIndex = 1;
%     for row = 1:patchSize:height
%         for col = 1:patchSize:width
%             % Ensure patch does not exceed image bounds
%             rowEnd = min(row + patchSize - 1, height);
%             colEnd = min(col + patchSize - 1, width);
% 
%             % Extract the image and truth patches
%             imgPatch = img(row:rowEnd, col:colEnd, :);
%             truthPatch = truth(row:rowEnd, col:colEnd);
% 
%             % Check if there are any positive pixels in the truth patch
%             if any(truthPatch(:))
%                 outputFolder = partOfLaneDir;
%             else
%                 outputFolder = notPartOfLaneDir;
%             end
% 
%             [~, file, ~] = fileparts(truthPath);
% 
%             % Save the patch
%             patchFileName = sprintf('%s_patch_%d.png', file, patchIndex);
%             patchPath = fullfile(outputFolder, patchFileName);
%             imwrite(imgPatch, patchPath);
%             patchIndex = patchIndex + 1;
%         end
%     end
% 
%     fprintf('Processing complete. Patches saved in %s and %s.\n', partOfLaneDir, notPartOfLaneDir);
% end
% 
% 
% % Define the dataset path
% base_path = "dataset\FixData";
% 
% % Get all JPG files
% image_files = dir(fullfile(base_path, "*.jpg"));
% 
% nfiles = 5;
% accuracies = zeros(1, nfiles); % Preallocate for efficiency
% 
% % Initialize counters and timers
% start_time = tic;
% last_500_time = tic;
% 
% %for j = 1 : nfiles
% for j = 1:nfiles
%     image_filename = image_files(j).name;
%     label_filename = strrep(image_filename, ".jpg", ".png");
% 
%     filename = fullfile(base_path, image_filename);
%     filename_label = fullfile(base_path, label_filename);
% 
%     if ~isfile(filename_label)
%         continue; % Skip if matching label file doesn't exist
%     end
% 
%     splitAndSortImages(filename, filename_label, 'minimumData/notResized');
% 
%     % Timing for every 500 images
%     if mod(j, 500) == 0
%         elapsed_500 = toc(last_500_time);
%         fprintf("Time for last 500 images: %.2f seconds\n", elapsed_500);
%         last_500_time = tic; % Reset the timer
%     end
% 
%     % Progress update
%     if mod(j, 1000) == 0 || j == nfiles
%         fprintf("Processed %d/%d images. Estimated time remaining: %.2f seconds.\n", j, nfiles, toc(start_time) / j * (nfiles - j));
%     end
% end
% 
% function resizeImages(inputFolder, outputFolder, targetSize)
%     % Get all image files from the input folder
%     imageFiles = dir(fullfile(inputFolder, '*.png')); % Change extension if needed
% 
%     % Create output folder if it doesn't exist
%     if ~exist(outputFolder, 'dir')
%         mkdir(outputFolder);
%     end
% 
%     % Process each image
%     for i = 1:length(imageFiles)
%         % Read the image
%         imgPath = fullfile(inputFolder, imageFiles(i).name);
%         img = imread(imgPath);
% 
%         % Resize the image
%         imgResized = imresize(img, targetSize);
% 
%         % Save the resized image
%         outputPath = fullfile(outputFolder, imageFiles(i).name);
%         imwrite(imgResized, outputPath);
%     end
% 
%     fprintf('Resized %d images in %s and saved to %s.\n', length(imageFiles), inputFolder, outputFolder);
% end
% 
% % Define paths
% inputDir1 = 'minimumData/notResized/partOfLane'; 
% inputDir2 = 'minimumData/notResized/notPartOfLane';
% 
% outputDir1 = 'minimumData/resized/partOfLane';
% outputDir2 = 'minimumData/resized/notPartOfLane';
% 
% % Set target size for AlexNet (227x227)
% targetSize = [227, 227];
% 
% % Resize both classes
% resizeImages(inputDir1, outputDir1, targetSize);
% resizeImages(inputDir2, outputDir2, targetSize);
% 
% output_folder = 'minimumData/resized';
% 
% imds = imageDatastore(output_folder, 'IncludeSubfolders', true, 'LabelSource', 'foldernames');
% countEachLabel(imds);
% [imdsTrain, imdsValidation] = splitEachLabel(imds, 0.8, 'randomized');
% 
% net = alexnet;
% layers = net.Layers;
% new_fc8 = fullyConnectedLayer(2, 'Name', 'fc8', 'WeightLearnRateFactor', 10, 'BiasLearnRateFactor', 10);
% layers(23) = new_fc8;
% layers(24) = softmaxLayer('Name', 'softmax');
% layers(25) = classificationLayer('Name', 'output');
% 
% options = trainingOptions('sgdm', ...
%     'MiniBatchSize', 16, ...
%     'MaxEpochs', 5, ...
%     'InitialLearnRate', 0.0001, ...
%     'Shuffle', 'every-epoch', ...
%     'ValidationData', imdsValidation, ...
%     'ValidationFrequency', 5, ...
%     'Verbose', true, ...
%     'Plots', 'training-progress');
% 
% trainedNet = trainNetwork(imdsTrain, layers, options);
% 
% predictedLabels = classify(trainedNet, imdsValidation);
% accuracy = mean(predictedLabels == imdsValidation.Labels);
% fprintf('Validation Accuracy: %.2f%%\n', accuracy * 100);
% 
% save('trainedAlexNet_LaneDetection.mat', 'trainedNet');

% load('trainedAlexNet_LaneDetection.mat', 'trainedNet'); % Load the trained network
% 
% imagePath = '000_Both_left_curve_0001_patch_70';
% img = imread(imagePath);
% 
% img = imresize(img, [227, 227]);
% 
% predictedLabel = classify(trainedNet, img);
% 
% imshow(img);
% title(['Predicted: ', char(predictedLabel)]);

% Load trained network
load('trainedAlexNet_LaneDetection.mat', 'trainedNet');

% Define image path
imagePath = 'C:\Users\beloremd\Desktop\CSSE463_Lane_Detection\dataset\FixData\000_Both_left_curve_0009.jpg';

% Define patch size
patchSize = 10;

% Generate lane mask
laneMask = getLaneMask(imagePath, patchSize, trainedNet);

% Display the results
figure;
subplot(1, 2, 1);
imshow(imread(imagePath));
title('Original Image');

subplot(1, 2, 2);
imshow(laneMask);
title('Lane Detection Mask');

