function [data, infoOut] = preprocessTrainingData(data, info, imageSize)
% Resize the training image and associated pixel label image.
data{1} = imresize(data{1}, imageSize);
data{2} = imresize(data{2}, imageSize);
infoOut = info;
end

%%

imageDir = "./dataset/split/train";
imds = imageDatastore(imageDir, 'FileExtensions', '.jpg');
labelDir = "./dataset/split/train";
classNames = ["lane","background"];
labelIDs   = {[1; 2; 3; 4; 5; 6; 7; 8; 9] 0};
pxds = pixelLabelDatastore(labelDir, classNames, labelIDs, 'FileExtensions', '.png');
imageSize = [256 256];
numClasses = numel(classNames);
cds = combine(imds, pxds);
tds = transform(cds, @(data, info)preprocessTrainingData(data, info, imageSize), 'IncludeInfo', true);
fprintf("Done\n");


net = deeplabv3plus(imageSize,numClasses,"resnet18");
%
checkpointPath = "./deeplab_checkpoints";
opts = trainingOptions("sgdm",...
    MiniBatchSize=8,...
    Plots="training-progress",...
    MaxEpochs=2, ...
    CheckpointPath=checkpointPath);

%
%%
load('deeplab_checkpoints/net_checkpoint__1592__2025_02_17__13_28_21.mat','net');
net = trainnet(tds,net,"crossentropy",opts);
%%
%net = trainnet(tds,net,"crossentropy",opts);

%%
save("./deepnet.mat", 'net');
fprintf("saved\n");

%%
testImg = imread('dataset/split/test/000_Both_left_curve_0009.jpg');
testImg = imresize(testImg, imageSize);
trueMask = imread('dataset/split/test/000_Both_left_curve_0009.png');
trueMask = logical(trueMask);
trueMask = imresize(trueMask, imageSize);
prediction = semanticseg(testImg, net);
predictionMask = zeros(size(prediction));
predictionMask(prediction == 'C1') = 1;
result = labeloverlay(testImg, prediction);

testImg2 = imread('dataset/split/test/000_Right_right_curve_3965.jpg');
testImg2 = imresize(testImg2, imageSize);
trueMask2 = imread('dataset/split/test/000_Right_right_curve_3965.png');
trueMask2 = logical(trueMask2);
trueMask2 = imresize(trueMask2, imageSize);
prediction2 = semanticseg(testImg2, net);
predictionMask2 = zeros(size(prediction2));
predictionMask2(prediction2 == 'C1') = 1;
result2 = labeloverlay(testImg2, prediction2);


figure;

subplot(2, 4, 1);
imshow(testImg);
title('Source Image');
subplot(2, 4, 2);
imshow(trueMask);
title('True Mask');
subplot(2, 4, 3);
imshow(predictionMask);
title('Prediction Mask');
subplot(2, 4, 4);
imshow(result);
title('Overlay on Original');

subplot(2, 4, 5);
imshow(testImg2);
title('Source Image');
subplot(2, 4, 6);
imshow(trueMask2);
title('True Mask');
subplot(2, 4, 7);
imshow(predictionMask2);
title('Prediction Mask');
subplot(2, 4, 8);
imshow(result2);
title('Overlay on Original');

%% 
testDir = "./dataset/split/test";
testImages = imageDatastore(testDir, 'FileExtensions', '.jpg');
testLabels = imageDatastore(testDir, 'FileExtensions', '.png');

numFiles = numel(testImages.Files);
% Initialize the array to store IoU, TPR, FPR, and Precision
metrics = cell(numFiles, 5); % numFiles x 5 array: columns for IoU, TPR, FPR, Precision, and image names

fprintf("Started evaluating...\n");
startTime = tic;
for i = 1:numFiles
    % Read test image and ground truth label
    testImg = imread(testImages.Files{i});
    testImg = imresize(testImg, imageSize);
    groundTruth = imread(testLabels.Files{i});
    groundTruth = imresize(groundTruth, imageSize);
    
    % Perform segmentation
    prediction = semanticseg(testImg, net);
    predMask = zeros(size(prediction));
    predMask(prediction == 'C1') = 1;
    trueMask = logical(groundTruth);

    % Compute the metrics: IoU, TPR, FPR, Precision
    [IoU, TPR, FPR, Precision] = computeMaskMetrics(predMask, trueMask);

    % Store the IoU score in the first column
    metrics{i, 1} = IoU;  % Store IoU score
    
    % Store the TPR in the second column
    metrics{i, 2} = TPR;  % Store True Positive Rate
    
    % Store the FPR in the third column
    metrics{i, 3} = FPR;  % Store False Positive Rate
    
    % Store the Precision in the fourth column
    metrics{i, 4} = Precision;  % Store Precision
    
    % Store the image name (without path) in the fifth column
    [~, imageName, ~] = fileparts(testImages.Files{i});  % Extract image name without path and extension
    metrics{i, 5} = imageName;  % Store image name

    if rem(i, 500) == 0 || i == 1
        elapsedTime = toc(startTime);
        avgTimePerIter = elapsedTime / i;
        remainingTime = avgTimePerIter * (numFiles - i);
        hours = floor(remainingTime / 3600);
        minutes = floor(mod(remainingTime, 3600) / 60);
        seconds = floor(mod(remainingTime, 60));

        fprintf("Finished with %d/%d. Elapsed %.2fs. Remaining: %02d:%02d:%02d\n", i, numFiles, elapsedTime, hours, minutes, seconds);
    end
end

%% 
% Count NaN values and set them to zero for IoU, TPR, FPR, Precision
nanCountIoU = sum(isnan(cell2mat(metrics(:, 1))));  % Count NaNs in the IoU column
nanCountTPR = sum(isnan(cell2mat(metrics(:, 2))));  % Count NaNs in the TPR column
nanCountFPR = sum(isnan(cell2mat(metrics(:, 3))));  % Count NaNs in the FPR column
nanCountPrecision = sum(isnan(cell2mat(metrics(:, 4))));  % Count NaNs in the Precision column

% Replace NaN values with zero for all metrics
metrics(isnan(cell2mat(metrics(:, 1))), 1) = {0};  % Set NaN IoU values to zero
metrics(isnan(cell2mat(metrics(:, 2))), 2) = {0};  % Set NaN TPR values to zero
metrics(isnan(cell2mat(metrics(:, 3))), 3) = {0};  % Set NaN FPR values to zero
metrics(isnan(cell2mat(metrics(:, 4))), 4) = {0};  % Set NaN Precision values to zero

fprintf("Number of NaN values - IoU: %d, TPR: %d, FPR: %d, Precision: %d\n", ...
    nanCountIoU, nanCountTPR, nanCountFPR, nanCountPrecision);

% Average metrics over test set
averageIoU = sum(cell2mat(metrics(:, 1))) / numFiles;
averageTPR = sum(cell2mat(metrics(:, 2))) / numFiles;
averageFPR = sum(cell2mat(metrics(:, 3))) / numFiles;
averagePrecision = sum(cell2mat(metrics(:, 4))) / numFiles;

fprintf("Average IoU: %.4f\n", averageIoU);
fprintf("Average TPR: %.4f\n", averageTPR);
fprintf("Average FPR: %.4f\n", averageFPR);
fprintf("Average Precision: %.4f\n", averagePrecision);
