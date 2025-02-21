load('final_checkpoint_deeplab.mat');
imageSize = [256 256];

%%%% Put image path below %%%%
testImg = imread('./dataset/split/test/000_Both_left_curve_0009.jpg');
testImg = imresize(testImg, imageSize);

prediction = semanticseg(testImg, net);
predMask = zeros(size(prediction));
predMask(prediction == 'C1') = 1;

figure;
subplot(1, 2, 1);
imshow(testImg);
title('Source Image');
subplot(1, 2, 2);
imshow(predMask);
title('Predicted Mask');
