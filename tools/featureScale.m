function [ x_norm ] = featureScale(x)
% scale x to [-1, 1]

    k = 2/(max(x) - min(x));
    x_norm = -1 + k * (x - min(x));
end

