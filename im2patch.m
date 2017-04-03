function Patch = im2patch(Im, Par)
Patch   =  zeros(Par.ps2ch, Par.lenrc, 'double');
k          =   0;
for l = 1:Par.ch
    for i  = 1:Par.ps
        for j  = 1:Par.ps
            k        =  k+1;
            blk  =  Im(Par.r-1+i, Par.c-1+j, l);
            Patch(k,:)  =  blk(:)';
        end
    end
end