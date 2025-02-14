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

%%
save("./deepnet.mat", 'net');

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
