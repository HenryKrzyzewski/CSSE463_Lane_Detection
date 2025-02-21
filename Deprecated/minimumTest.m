% %% Load trained network
% load('trainedAlexNet_LaneDetection.mat', 'trainedNet');
% 
% %% Define image path
% imagePath = "dataset\split\test\000_Both_straight_4138.jpg";
% labelPath = "dataset\split\test\000_Both_straight_4138.png";
% labelMask = imbinarize(imread(labelPath));
% % Define patch size
% patchSize = 10;
% 
% % Generate lane mask
% laneMask = getLaneMask(imagePath, patchSize, trainedNet);
% 
% %% post processing
% laneMaskModified = laneMask;
% 
% SE = strel('disk', 12);         % Create a 10x10 square structuring element
% laneMaskModified = imclose(laneMaskModified, SE); % Apply morphological closing
% laneMaskModified = bwareaopen(laneMaskModified, 1400);
% % Display the results
% figure;
% subplot(2, 2, 1);
% imshow(imread(imagePath));
% title('Original Image');
% 
% subplot(2, 2, 2);
% imshow(labelMask);
% title('Lane Detection label Mask');
% 
% subplot(2, 2, 3);
% imshow(laneMask);
% title('Lane Detection Mask');
% 
% subplot(2, 2, 4);
% imshow(laneMaskModified);
% title('Lane Detection Mask Modified');
% 
% [IoU, TPR, FPR, Precision] = computeMaskMetrics(laneMask, labelMask);
% [IoUModified, TPRModified, FPRModified, PrecisionModified] = computeMaskMetrics(laneMaskModified, labelMask);
% fprintf('Elapsed time: %.2f seconds\n', toc); % Print elapsed time

function laneDetection(filePath)
    %% Load trained network
    load('trainedAlexNet_LaneDetection.mat', 'trainedNet');
    
    %% Define image and label paths
    imagePath = filePath;
    labelPath = strrep(filePath, '.jpg', '.png');
    labelMask = imbinarize(imread(labelPath));
    
    % Define patch size
    patchSize = 10;
    
    % Generate lane mask
    laneMask = getLaneMask(imagePath, patchSize, trainedNet);
    
    %% Post-processing
    laneMaskModified = laneMask;
    SE = strel('disk', 12);
    laneMaskModified = imclose(laneMaskModified, SE);
    laneMaskModified = bwareaopen(laneMaskModified, 1400);
    
    %% Display results
    figure;
    subplot(2, 2, 1);
    imshow(imread(imagePath));
    title('Original Image');
    
    subplot(2, 2, 2);
    imshow(labelMask);
    title('Lane Detection Label Mask');
    
    subplot(2, 2, 3);
    imshow(laneMask);
    title('Lane Detection Mask');
    
    subplot(2, 2, 4);
    imshow(laneMaskModified);
    title('Lane Detection Mask Modified');
    
    %% Compute Metrics
    [IoU, TPR, FPR, Precision] = computeMaskMetrics(laneMask, labelMask);
    [IoUModified, TPRModified, FPRModified, PrecisionModified] = computeMaskMetrics(laneMaskModified, labelMask);
    
    fprintf('IoU: %.2f, TPR: %.2f, FPR: %.2f, Precision: %.2f\n', IoU, TPR, FPR, Precision);
    fprintf('IoU (Modified): %.2f, TPR (Modified): %.2f, FPR (Modified): %.2f, Precision (Modified): %.2f\n', IoUModified, TPRModified, FPRModified, PrecisionModified);
end

laneDetection("dataset\split\test\000_Both_straight_4138.jpg");