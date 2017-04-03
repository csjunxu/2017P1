clear,clc;
addpath('../Utilities/');
%% data
trainPATH = 'C:/Users/csjunxu/Desktop/JunXu/Datasets/The Berkeley Segmentation Dataset/BSR_full/BSR/BSDS500/data/images/trainpng/';
testCleanPATH = 'C:/Users/csjunxu/Desktop/JunXu/Datasets/The Berkeley Segmentation Dataset/BSR_full/BSR/BSDS500/data/images/valpng';
testNoisyPATH = '../../TrainingData/ycbcrNoisy/';

%% loading GMM trained in 'collectPatches.m'
patch_size=8; 
num_patch_N = 100000;
num_patch_C = 50*num_patch_N;
R_thresh = 0.05;
load ../Data/EMGM_8x8_100_knnNI2BS500Train_20160722T082406.mat;
GMM = model;

%% discriminative training of gating network
% network architechture 
input_shape = [patch_size^2,1,1,1];
lspec={};
lspec{1} = struct('type','affine',  'out_shape',50);
lspec{2} = struct('type','square');
lspec{3} = struct('type','affine',  'out_shape',100);
net = neuNet(lspec, input_shape);
% minibatches of 10K patches
getData = @() loadImagePatches(trainPATH,[patch_size,patch_size],50000)/255;
[testNoisyData, testCleanData] = rnd_smp_patch_kNN(testNoisyPATH, testCleanPATH, patch_size, num_patch_N, num_patch_C, R_thresh);
% train Gating network using the cross-entropy loss on 500 minibatches 
net = trainGMMGatingNet_CN(GMM,net,getData,testCleanData,testNoisyData,100,'GatingNet_knnNI2BS500_5e4_1e5_100'); 