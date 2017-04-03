%% load pre-trained models (see demoTrain.m for training models)
GMM = getfield(load('GMM200.mat'),'model');
GMM.net = getfield(load('gating100.mat'),'net');

%% image denoising

% load image 
% origI = mean(double(imread('123074.jpg')),3)/255;
origI = rgb2gray(double(imread('123074.jpg'))/255);
% add noise
noiseVar = (25/255)^2;
noisyI = origI + sqrt(noiseVar)*randn(size(origI));

% denoise params
stride = 3; % change for speed/accuracy tradeoff (might require update of lambda)
lambdas = GMM.p^2/noiseVar/stride^2*[0.7,0.6,0.5,0.4];
betas = [1,5,10,50]/noiseVar;
% GMM patch denoising function handle
patchMAP = @(Zi,noiseVar) MAPGMM(Zi,Zi,GMM,noiseVar,struct('T',3, 'hardEM',1, 'calcCost',0)); 


% denoise 
tic;
resI = denoiseEPLL(noisyI,[GMM.p,GMM.p],patchMAP,lambdas,betas,stride); 
t=toc;

PSNR = 20*log10(1./std2(resI-origI));
fprintf('time %f PSNR %f\n',t,PSNR);
figure(1); imshow(origI); title('original');
figure(2); imshow(noisyI); title('noisy');
figure(3); imshow(resI); title('denoised');