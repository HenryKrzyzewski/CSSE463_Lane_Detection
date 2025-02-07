function [IoU, TPR, FPR, Precision] = computeMaskMetrics(testMask, truthMask)
    % computeStats calculates the Intersection over Union (IoU), True Positive Rate (TPR),
    % False Positive Rate (FPR), and Precision between two binary images.
    % Inputs:
    %   testMask - First binary image (logical matrix)
    %   truthMask - Second binary image (logical matrix)
    % Outputs:
    %   IoU - Intersection over Union value
    %   TPR - True Positive Rate
    %   FPR - False Positive Rate
    %   Precision - Positive Predictive Value

    if ~isequal(size(testMask), size(truthMask))
        error('Input images must have the same dimensions');
    end
    
    % Compute intersection and union
    intersection = testMask & truthMask;
    union = testMask | truthMask;
    
    % Compute IoU
    IoU = sum(intersection(:)) / sum(union(:));
    
    % Compute True Positive Rate (TPR)
    truePositives = sum(intersection(:));
    actualPositives = sum(truthMask(:));
    TPR = truePositives / actualPositives;
    
    % Compute False Positive Rate (FPR)
    falsePositives = sum(testMask(:)) - truePositives;
    actualNegatives = sum(~truthMask(:));
    FPR = falsePositives / actualNegatives;
    
    % Compute Precision
    predictedPositives = sum(testMask(:));
    Precision = truePositives / predictedPositives;
end