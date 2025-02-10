function [data, infoOut] = preprocessTrainingData(data, info, imageSize)
% Resize the training image and associated pixel label image.
data{1} = imresize(data{1}, imageSize);
data{2} = imresize(data{2}, imageSize);

% Convert grayscale input image into RGB for use with ResNet-18, which
% requires RGB image input.
data{1} = repmat(data{1},1,1,3);

infoOut = info;

assignin('base', 'thedemodata', data);
assignin('base', 'thedemoinfo', info);
end

%%
dataSetDir = fullfile(toolboxdir("vision"),"visiondata","triangleImages");
imageDir = fullfile(dataSetDir,"trainingImages");
imds = imageDatastore(imageDir);
labelDir = fullfile(dataSetDir, "trainingLabels");
classNames = ["triangle","background"];
labelIDs   = [255 0];
pxds = pixelLabelDatastore(labelDir,classNames,labelIDs);
imageSize = [256 256];
numClasses = numel(classNames);
cds = combine(imds,pxds);
tds = transform(cds, @(data, info)preprocessTrainingData(data,info,imageSize), 'IncludeInfo', true);
fprintf("Done\n");

%%
net = deeplabv3plus(imageSize,numClasses,"resnet18");
opts = trainingOptions("sgdm",...
    MiniBatchSize=8,...
    MaxEpochs=3);

net = trainnet(tds,net,"crossentropy",opts);
