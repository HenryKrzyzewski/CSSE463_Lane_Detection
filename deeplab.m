%%
imagesize = [256 256];
numClasses = 2;
networkName = "resnet18";
net = deeplabv3plus(imagesize, numClasses, networkName);
analyzeNetwork(net);

%%
function data = preprocTrainingData(data, imsz)
    data{1} = imresize(data{1}, imsz);
    data{2} = imresize(data{2}, imsz);
end

imds = imageDatastore("./dataset/split/train", 'FileExtensions', '.jpg');
classNames = ['lane', 'background'];
labelIds = [255 0];
pxds = pixelLabelDatastore("./dataset/split/train", classNames, labelIds, 'FileExtensions', '.png');
cds = combine(imds, pxds);
tds = transform(cds, @(data)preprocTrainingData(data,imagesize));

opts = trainingOptions("sgdm",...
    MiniBatchSize=8,...
    MaxEpochs=3);
fprintf('Done\n');

%%
net = trainnet(tds,net,"crossentropy",opts);
