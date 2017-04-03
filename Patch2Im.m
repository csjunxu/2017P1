function Im = Patch2Im(X, Par)
% Reconstruction
Im = zeros(Par.h, Par.w, Par.ch);
Wei = zeros(Par.h, Par.w, Par.ch);
r = 1:Par.step:Par.maxr;
c = 1:Par.step:Par.maxc;
k = 0;
for l = 1:1:Par.ch
    for i = 1:1:Par.ps
        for j = 1:1:Par.ps
            k = k+1;
            Im(Par.r-1+i, Par.c-1+j, l)    =  Im(Par.r-1+i, Par.c-1+j, l) + reshape( X(k,:)', [Par.lenr Par.lenc]);
            Wei(Par.r-1+i, Par.c-1+j, l)  =  Wei(Par.r-1+i, Par.c-1+j, l) + 1;     
        end
    end
end
Im  =  Im./(Wei + eps);