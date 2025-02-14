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

%%
net = deeplabv3plus(imageSize,numClasses,"resnet18");
opts = trainingOptions("sgdm",...
    MiniBatchSize=8,...
    Plots="training-progress",...
    MaxEpochs=3);

%%
net = trainnet(tds,net,"crossentropy",opts);
save('./deepnet.mat', net);

%%
testImg = imread('dataset/split/test/000_Both_left_curve_0009.jpg');
testImg = imresize(testImg, imageSize);
prediction = semanticseg(testImg, net);
predictionMask = zeros(size(prediction));
predictionMask(prediction == 'C2') = 1;
result = labeloverlay(testImg, prediction);
%%
figure;
subplot(1, 3, 1);
imshow(testImg);
subplot(1, 3, 2);
imshow(predictionMask);
subplot(1, 3, 3);
imshow(result);


%% 
testDir = "./dataset/split/test";
testImages = imageDatastore(testDir, 'FileExtensions', '.jpg');
testLabels = pixelLabelDatastore(testDir, classNames, labelIDs, 'FileExtensions', '.png');

tdsTest = combine(testImages, testLabels);
iouScores = zeros(numel(classNames), 1);

for i = 1:numel(tdsTest.Files)
    % Read test image and ground truth label
    data = read(tdsTest);
    testImg = imresize(data{1}, imageSize);
    groundTruth = imresize(uint8(data{2}), imageSize, 'nearest');
    
    % Perform segmentation
    prediction = semanticseg(testImg, net);
    predictionMask = zeros(size(prediction));
    for c = 1:numClasses
        predictionMask(strcmp(prediction, classNames{c})) = c;
    end
    
    % Compute IoU for each class
    for c = 1:numClasses
        predMask = predictionMask == c;
        gtMask = groundTruth == c;
        
        intersection = sum(predMask(:) & gtMask(:));
        union = sum(predMask(:) | gtMask(:));
        
        if union > 0
            iouScores(c) = iouScores(c) + (intersection / union);
        end
    end
end

% Average IoU over test set
iouScores = iouScores / numel(tdsTest.Files);

disp("Intersection over Union for each class:");
for c = 1:numClasses
    fprintf("%s: %.4f\n", classNames(c), iouScores(c));
end

