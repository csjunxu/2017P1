function net = trainGMMGatingNet_CN(GMM,net,getData,testCleanData,testNoisyData,niter,filename)
% train a GMM gating net with cross entropy loss
% GMM - the generative referece model
% net - initial network
% getData - function handle to get minibatches
% testCleanData - the clean batch for testing
% testNoisyData - the noisy batch for testing
% niter - number of iterations (minibatches)
% filename - file name to store resulting network
%
% Used in the paper:
% "The Return of the Gating Network: Combining Generative Models and Discriminative
% Training in Natural Image Priors" by Dan Rosenbaum and Yair Weiss
%
% Dan Rosenbaum http://www.cs.huji.ac.il/~danrsm
%

miniters = 100; % itreations of minimize for each minibatch
% parameters of MAP computation (for testing denoising)
paramMAP=struct('T',3, 'hardEM',1, 'calcCost',0);

W = getNetworkWeights(net.theta);
GMM.net = net;

Xtest = testCleanData;
Ytest = testNoisyData;
% Noise Level Estimation
noiseVar = NoiseLevel(Ytest*255); % can be improved
fprintf('The noise level is %2.4f.\n',noiseVar);
noiseVar = (noiseVar/255)^2;

% calc exact posterior probability
logPtest = zeros(GMM.nmodels,size(Xtest,2));
for i=1:GMM.nmodels
    logPtest(i,:) = log(GMM.mixweights(i)) + loggausspdf(Xtest,GMM.covs(:,:,i));
end
Ptest = exp(bsxfun(@minus,logPtest,logsumexp(logPtest)));

% denoise with full GMM (without gating)
Xhat_test0 = MAPGMM(Ytest,Ytest,rmfield(GMM,'net'),noiseVar,paramMAP,Xtest);
PSNR0 = 20*log10(1./std2(Xhat_test0-Xtest));

% iterate over minibatches
Ltrain = zeros(1,niter);
Ltest = zeros(1,niter);
PSNR = zeros(1,niter);
for b=1:niter
    Xtrain = getData();
    
    % calc exact posterior probability
    logP = zeros(GMM.nmodels,size(Xtrain,2));
    for i=1:GMM.nmodels
        logP(i,:) = log(GMM.mixweights(i)) + loggausspdf(Xtrain,GMM.covs(:,:,i));
    end
    P = exp(bsxfun(@minus,logP,logsumexp(logP)));
    
    Ltrain(b) = crossEntropyLoss(W,net,Xtrain,P);
    Ltest(b) = crossEntropyLoss(W,net,Xtest,Ptest);
    Xhat_test = MAPGMM(Ytest,Ytest,GMM,noiseVar,paramMAP,Xtest);
    PSNR(b) = 20*log10(1./std2(Xhat_test-Xtest));
    fprintf('minibatch %d trainXE %f testXE %f gatingPSNR %f fullPSNR %f\n',b,Ltrain(b),Ltest(b),PSNR(b),PSNR0);
    W = minimize(W(:),@crossEntropyLoss,-miniters,net,Xtrain,P);
    net.theta = setNetworkWeights(W,GMM.net.theta);
    GMM.net = net;
    if exist('filename','var'), save(filename,'net','Ltrain','Ltest','PSNR'); end
end
