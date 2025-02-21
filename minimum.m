
%% Function: Split and Sort Images into Categories
function splitAndSortImages(imagePath, truthPath, outputDir)
    img = imread(imagePath);
    truth = imbinarize(imread(truthPath)); % Ensure binary truth image
    [height, width, ~] = size(img);
    patchSize = 20;

    % Create output directories if they don't exist
    partOfLaneDir = fullfile(outputDir, 'partOfLane');
    notPartOfLaneDir = fullfile(outputDir, 'notPartOfLane');
    if ~exist(partOfLaneDir, 'dir'), mkdir(partOfLaneDir); end
    if ~exist(notPartOfLaneDir, 'dir'), mkdir(notPartOfLaneDir); end

    % Iterate over the image in patches
    patchIndex = 1;
    for row = 1:patchSize:height
        for col = 1:patchSize:width
            rowEnd = min(row + patchSize - 1, height);
            colEnd = min(col + patchSize - 1, width);
            imgPatch = img(row:rowEnd, col:colEnd, :);
            truthPatch = truth(row:rowEnd, col:colEnd);

            % Assign patches to categories based on label presence
            outputFolder = partOfLaneDir;
            if ~any(truthPatch(:)), outputFolder = notPartOfLaneDir; end
            patchFileName = sprintf('%s_patch_%d.png', fileparts(truthPath), patchIndex);
            imwrite(imgPatch, fullfile(outputFolder, patchFileName));
            patchIndex = patchIndex + 1;
        end
    end
    fprintf('Patches saved in %s and %s.\n', partOfLaneDir, notPartOfLaneDir);
end

%% Process and Sort Dataset
base_path = "dataset/FixData";
image_files = dir(fullfile(base_path, "*.jpg"));
image_files = image_files(randperm(length(image_files))); % Shuffle dataset
nfiles = 5; % Number of files to process

for j = 1:nfiles
    image_filename = image_files(j).name;
    label_filename = strrep(image_filename, ".jpg", ".png");
    filename = fullfile(base_path, image_filename);
    filename_label = fullfile(base_path, label_filename);
    if isfile(filename_label)
        splitAndSortImages(filename, filename_label, 'minimumData/notResized');
    end
end

%% Function: Resize Images
function resizeImages(inputFolder, outputFolder, targetSize)
    imageFiles = dir(fullfile(inputFolder, '*.png'));
    if ~exist(outputFolder, 'dir'), mkdir(outputFolder); end
    for i = 1:length(imageFiles)
        img = imresize(imread(fullfile(inputFolder, imageFiles(i).name)), targetSize);
        imwrite(img, fullfile(outputFolder, imageFiles(i).name));
    end
    fprintf('Resized images saved in %s.\n', outputFolder);
end

%% Resize Dataset for AlexNet
resizeImages('minimumData/notResized/partOfLane', 'minimumData/resized/partOfLane', [227, 227]);
resizeImages('minimumData/notResized/notPartOfLane', 'minimumData/resized/notPartOfLane', [227, 227]);

%% Prepare Data for Training
imds = imageDatastore('minimumData/resized', 'IncludeSubfolders', true, 'LabelSource', 'foldernames');
[imdsTrain, imdsValidation] = splitEachLabel(imds, 0.8, 'randomized');

%% Modify and Train AlexNet
net = alexnet;
layers = net.Layers;
layers(23) = fullyConnectedLayer(2, 'Name', 'fc8', 'WeightLearnRateFactor', 10, 'BiasLearnRateFactor', 10);
layers(24) = softmaxLayer('Name', 'softmax');
layers(25) = classificationLayer('Name', 'output');

options = trainingOptions('sgdm', 'MiniBatchSize', 16, 'MaxEpochs', 5, 'InitialLearnRate', 0.0001, 'Shuffle', 'every-epoch', 'ValidationData', imdsValidation, 'ValidationFrequency', 5, 'Verbose', true, 'Plots', 'training-progress');

trainedNet = trainNetwork(imdsTrain, layers, options);
save('trainedAlexNet_LaneDetection.mat', 'trainedNet');

%% Lane Detection Function
function laneDetection(filePath)
    load('trainedAlexNet_LaneDetection.mat', 'trainedNet');
    imagePath = filePath;
    labelMask = imbinarize(imread(strrep(filePath, '.jpg', '.png')));
    laneMask = getLaneMask(imagePath, 10, trainedNet);
    laneMaskModified = bwareaopen(imclose(laneMask, strel('disk', 12)), 1400);

    % Display results
    figure;
    subplot(2,2,1); imshow(imread(imagePath)); title('Original Image');
    subplot(2,2,2); imshow(labelMask); title('Lane Label Mask');
    subplot(2,2,3); imshow(laneMask); title('Lane Mask');
    subplot(2,2,4); imshow(laneMaskModified); title('Modified Lane Mask');
    
    % Compute metrics
    [IoU, TPR, FPR, Precision] = computeMaskMetrics(laneMask, labelMask);
    [IoUModified, TPRModified, FPRModified, PrecisionModified] = computeMaskMetrics(laneMaskModified, labelMask);
    fprintf('IoU: %.2f, TPR: %.2f, FPR: %.2f, Precision: %.2f\n', IoU, TPR, FPR, Precision);
    fprintf('IoU (Modified): %.2f, TPR (Modified): %.2f, FPR (Modified): %.2f, Precision (Modified): %.2f\n', IoUModified, TPRModified, FPRModified, PrecisionModified);
end

laneDetection("dataset/split/test/000_Both_straight_4138.jpg");
