%% image deblurring

%% load pre-trained models (see demoTrain.m for training models)
GMM = getfield(load('GMM200.mat'),'model');
GMM.net = getfield(load('gating100.mat'),'net');


% load image
origI = mean(double(imread('123074.jpg')),3)/255;

% blur the image
noiseVar = (2.5/255)^2;
K = fspecial('motion',10,45);
y = conv2(origI, K, 'valid');
y = y + sqrt(noiseVar)*randn(size(y));
y = double(uint8(y .* 255))./255;
ks = floor((size(K, 1) - 1)/2);
y = padarray(y, [1 1]*ks, 'replicate', 'both');
for aa=1:4, y = edgetaper(y, K); end
blurI = y;
    
% deblur params
stride = 7;
lambdas = GMM.p^2/noiseVar/stride^2;
betas = 30*[1,2,4,16,64,128,256];
patchMAP = @(Zi,noiseVar) MAPGMM(Zi,Zi,GMM,noiseVar,struct('T',2, 'hardEM',1, 'calcCost',0)); 


% deblur
tic;
resI = deblurEPLL(blurI, [GMM.p,GMM.p], patchMAP, K, lambdas, betas, stride);
t=toc;

PSNR = 20*log10(1./std2(resI-origI));
fprintf('time %f PSNR %f\n',t,PSNR);
figure(1); imshow(origI); title('original');
figure(2); imshow(blurI); title('blurred');
figure(3); imshow(resI); title('deblurred');

