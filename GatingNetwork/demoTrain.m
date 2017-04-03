clear,clc;
addpath('../Utilities/');
%% data
trainPATH = 'C:/Users/csjunxu/Desktop/JunXu/Datasets/The Berkeley Segmentation Dataset/BSDS300/images/train/';
testPATH = 'C:/Users/csjunxu/Desktop/JunXu/Datasets/The Berkeley Segmentation Dataset/BSDS300/images/test/';
% testPATH = '../TrainingImages/';
%% generative training of GMM
param=[];
param.p=8; % patch size
param.mbnumber=200; % number of minibatches (iterations)
param.reg=1e-6; % regularization of covariance
param.draw=1; % draw results
% generate minibatches by loading random patches from all images 
getData = @() loadImagePatches(trainPATH,'',[param.p,param.p],100000)/255; % trainPATH
% train a 200 component GMM and save it to 'tmpModel.mat'
GMM = trainGMM(100,getData,param,'tmpGMM');
% load tmpGMM.mat;
% GMM = model;

%% discriminative training of gating network
% network architechture
input_shape = [param.p^2,1,1,1];
lspec={};
lspec{1} = struct('type','affine',  'out_shape',50);
lspec{2} = struct('type','square');
lspec{3} = struct('type','affine',  'out_shape',100);
net = neuNet(lspec, input_shape);
% minibatches of 10K patches
getData = @() loadImagePatches(trainPATH,[param.p,param.p],10000)/255;
testData = loadImagePatches(testPATH,[param.p,param.p],100000)/255;
% train Gating network using the cross-entropy loss on 500 minibatches 
net = trainGMMGatingNet(GMM,net,getData,testData,500,'tmpGatingNet_10'); 