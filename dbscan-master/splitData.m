load 'all_2.mat'

[M,N] = size(mappedX);

ff_g = mappedX(1:11579,:);
ff_q = mappedX(11580:M,:);
