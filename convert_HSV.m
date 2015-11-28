%% feature2rgb: assumes a 1x3072 matrix,
%% first 1024 red, next 1024 green, final 1024 blue
function [hsv] = convert_HSV(feat, feature_parameters)
    r = double(feat(1:1024)) ./ 255;
    g = double(feat(1025:2048)) ./ 255;
    b = double(feat(2049:3072)) ./ 255;
    hsv = uint8(round(rgb2hsv([r' g' b']) .* 255));
    hsv = [hsv(:,1)' hsv(:,2)' hsv(:,3)'];
end
