function [ X, y, G ] = generate_data1(K)
% generate data for the overlapping group lasso experiment
% 
% Input:
%       n   number of samples
%       d   dimension
%       K   number of groups
% Output:
%       X   traing data, nxd
%       y   label dx1

    d = 90*K + 10;
    n = d;
    X = normrnd(0, 1, [n, d]);
    
    i = 1:d; i = i';
    w = (-1).^i .* exp(-(i-1)/100);
    
    y = X * w + normrnd(0, 1, [d, 1]);
    
    % generate the group cell 
    G = cell(K, 1);
    for j = 1:K
        G{j} = (90*j - 89) : (90*j + 10);
        G{j} = G{j}';
    end
end