function binaryMask = getLaneMask(imagePath, patchSize, net)
    % get input image
    img = imread(imagePath);
    [imgHeight, imgWidth, ~] = size(img);
    
    % Initialize binary mask
    binaryMask = zeros(imgHeight, imgWidth, 'logical');
    
    % Loop through patches in image
    for row = 1:patchSize:imgHeight
        for col = 1:patchSize:imgWidth
            rowEnd = min(row + patchSize - 1, imgHeight);
            colEnd = min(col + patchSize - 1, imgWidth);
            
            imgPatch = img(row:rowEnd, col:colEnd, :);
            imgPatch = imresize(imgPatch, [227, 227]); 
        
            predictedLabel = classify(net, imgPatch);
            
            if predictedLabel == "partOfLane"
                binaryMask(row:rowEnd, col:colEnd) = 1; 
            end
        end
    end
    
    % Convert logical mask to uint8 image
    binaryMask = uint8(binaryMask) * 255;
end