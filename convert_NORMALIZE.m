function [out] = convert_NORMALIZE(feat, feature_parameters)
  out = double(feat)/max(feat);
end