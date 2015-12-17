function feat = convert_HOG(image, feature_parameters)
%% Description
% This function takes one image as input and returns HOG feature.
%
% Input: image
% Following VLFeat instruction, the input image should be SINGLE precision.
% If not, the image is automatically converted to SINGLE precision.
%
% Output: feat
% The output is a vectorized HOG descriptor.
% The feature dimension depends on the parameter, cellSize.
%
% VLFeat must be added to MATLAB search path. Please check the link below.
% http://www.vlfeat.org/install-matlab.html

R = reshape(image(1:1024),32,32);
G = reshape(image(1+1024:2*1024),32,32);
B = reshape(image(1+2*1024:3*1024),32,32);
final_img(:,:,1) = R;
final_img(:,:,2) = G;
final_img(:,:,3) = B;

%% check input data type
if ~isa(final_img, 'single'), final_img = single(final_img); end;

if length(feature_parameters) == 0
    cellSize = 8;
else
    cellSize = feature_parameters{1};
end

%% extract HOG
hog = vl_hog(final_img, cellSize);
% imhog = vl_hog('render', hog);
% clf; imagesc(imhog); colormap gray;


%% feature - vectorized HOG descriptor
feat = hog(:);
end
