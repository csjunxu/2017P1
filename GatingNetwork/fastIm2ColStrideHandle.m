function [i2c,c2i,i2cidx,pcount] = fastIm2ColStrideHandle(imsize,psize,stride,N)
% Returns handle to functions similar to im2col and col2im.
%
% Inputs:
% imsize - [n,m,c] size of input image (can have multiple channels)
% psize - [pn,pm] size of patch
% stride - stride (skips stride-1 patches in each dimension)
% N - returns N handles each with a different offset (when stride>1)
%
% Outputs:
% i2c - given the image, returns patches.
% c2i - builds the average image of possibly overlapping patches
% i2cidx - the indices use for i2c and c2i
% pcount - the number of patches containing each pixel in the image
%
% Dan Rosenbaum http://www.cs.huji.ac.il/~danrsm
%

if (length(imsize)==3), channels=imsize(3); imsize=imsize(1:2); 
else channels = 1; end;
if (~exist('stride','var')), stride=1; end;
if (~exist('N','var')), N=1; end;

% create patch index map template
[px,py]=meshgrid(1:psize(2),1:psize(1));
pidx = py(:)+(px(:)-1)*imsize(1);

% create N index maps with different offsets
i2cidx=cell(N,1);
i2cCell=cell(N,1);
c2iCell=cell(N,1);
pcountCell=cell(N,1);
for n=1:N
    
    % modify according to stride and some random offset
    offset = [randi(stride),randi(stride)];
    yidx = [0,(offset(1)-1):stride:(imsize(1)-psize(1)-1),imsize(1)-psize(1)];
    xidx = permute([0,(offset(2)-1):stride:(imsize(2)-psize(2)-1),imsize(2)-psize(2)],[1,3,2])*imsize(1);
    i2cidx{n} = zeros(length(pidx),length(yidx),length(xidx));
    i2cidx{n} = reshape(bsxfun(@plus,bsxfun(@plus,bsxfun(@plus,i2cidx{n},pidx),yidx),xidx),[length(pidx),length(yidx)*length(xidx)]);

            
    % create function handle
    i2cCell{n} = @(im) (im(i2cidx{n}));

    if (nargout<2), continue; end
    
    % calc inverse mapping
    [sidx,inv_idx] = sort(i2cidx{n}(:));
    sum_idx = [find(sidx(2:end)~=sidx(1:end-1))',length(sidx)];
    sidx = unique(sidx);
    pcountCell{n}=[sum_idx(1),sum_idx(2:end)-sum_idx(1:end-1)];
    c2iCell{n} = @(X) (fastCol2Im(X,sidx,inv_idx,sum_idx,pcountCell{n},[imsize,channels]));
    pcountCell{n} = reshape(pcountCell{n},[imsize,channels]);
end

i2c = @(im,n) i2cCell{min(n,N)}(im);
c2i = @(im,n) c2iCell{min(n,N)}(im);
pcount = @(n) pcountCell{min(n,N)};

% use inverse mapping for 'average' col2im 
% based on code by Shai Shalev-Shwartz
function im=fastCol2Im(X,sidx,inv_idx,sum_idx,pcount,ressize)
cs=cumsum(X(inv_idx));
sumvec=cs(sum_idx)'-[0,cs(sum_idx(1:end-1))'];
im(sidx) = sumvec./pcount;
im = reshape(im,ressize);

