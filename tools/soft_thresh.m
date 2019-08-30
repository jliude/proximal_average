function [ w ] = soft_thresh(v, t)
    %v = max( 0, x - t*opts.rho ) - max( 0, -x - t*opts.rho );

    w = sign(v) .* max(abs(v) - t,0);
end

