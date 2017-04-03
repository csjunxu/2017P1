function [resI] = deblurEPLL(Y, psize, patchMAP, K, lambdas, betas,stride)
% Deblur the image 'Y' by minimizing the EPLL cost:
% J(X) = - sum_i logPr(X_i) + \lambda/2 ||A*X-Y||^2 
% where X is the image, Y is the given blurred image, A is a matrix
% implementation of the blurring (convolving K), and X_i are all the image patches.
%
% The cost is minimized by minimizing the following relaxation (penalty method):
% J(X,Z_i) = -sum_i logPr(Z_i) + lambda/2 ||A*X-Y||^2 + beta/2 sum_i||X_i-Z_i||^2
% where Z_i are allowed to be independent of each other but are gradually
% forced to equal X_i by increasing beta.
%
% In each iteration the problem solved by 
% (1) minimizing for each Z_i: -logPr(Z_i) + beta/2 ||X_i-Z_i||^2
%     (using the given patchMAP function handle).
% (2) minimize for X: lambda/2 ||X-Y||^2 + beta/2 sum_i||A*X_i-Z_i||^2
%     (by averaging all Z_i and Y).
% 
% input:
% Y - the noisy image
% psize - the patch size
% patchMAP - a patch denoising function handle, performing (approximate) MAP.
% K - the blur kernel
% lambdas - lambda in the cost (could be more than one value to allow it to
%           change through the iterations.
% betas - beta values for the different iterations (determins the # of iterations).
% stride - if bigger than 1, only a subset of patches are used (default=1).
%
% Used in the paper: 
% "The Return of the Gating Network: Combining Generative Models and Discriminative 
% Training in Natural Image Priors" by Dan Rosenbaum and Yair Weiss
% 
% Dan Rosenbaum http://www.cs.huji.ac.il/~danrsm
%

if ~exist('stride','var'), stride=1; end

% init im2col/col2im handles
imsize = size(Y);
[i2c,c2i,~,pcount] = fastIm2ColStrideHandle(imsize,psize,stride,min(stride^2,length(betas)));

% init data
Y = Y - 0.5;
Aty = Y(floor(size(K,1)/2)+1:end-floor(size(K,1)/2),floor(size(K,2)/2)+1:end-floor(size(K,2)/2));
Aty = conv2(Aty,rot90(rot90(K)),'full');
X = Y;

for i=1:length(betas)
    beta = betas(i);  
    lambda = lambdas(min(i,length(lambdas)));
    
    % minimize (1)
    Zi = patchMAP(i2c(X,i),1/beta);
    
    % minimize (2) 
    XZ = lambda*Aty+beta*pcount(i).*c2i(Zi,i);
    [X,~] = bicg(@(x,tf) Afun(x,K,beta*pcount(i),lambda,imsize),XZ(:),1e-3,100,[],[],X(:));
    X = reshape(X,imsize);
    
end

resI =  X + 0.5;

% function to apply the corruption model (implemented efficiantly using
% convolutions instead of matrix multiplications)
function y = Afun(x,K,betaCounts,lambda,ss)
x = reshape(x,ss);
tt = imfilter(x,K,'conv','same');
tt = tt(floor(size(K,1)/2)+1:end-floor(size(K,1)/2),floor(size(K,2)/2)+1:end-floor(size(K,2)/2));
y = lambda*imfilter(tt,rot90(rot90(K)),'conv','full');
y = y(:) + betaCounts(:).*x(:);
