% This function collect the paired patches in corresponding
% 1. Noisy and Clean
% 2. Mosaicked and Demosaicked
% 3. Low Resolution and High Resolution
% Images


clear;

TT_Original_image_dir = 'C:/Users/csjunxu/Desktop/Projects/RID_Dataset/20170121/Canon_80D_ISO800/';
GT_Original_image_dir = 'C:/Users/csjunxu/Desktop/Projects/RID_Dataset/20170121/Canon_80D_ISO800mean/';
GT_fpath = fullfile(GT_Original_image_dir, '*.JPG');
TT_fpath = fullfile(TT_Original_image_dir, '*.JPG');

GT_im_dir  = dir(GT_fpath);
TT_im_dir  = dir(TT_fpath);
im_num = length(TT_im_dir);

S = regexp(TT_Original_image_dir, '/', 'split');
IMname = S{end-1};

Par.ps = 6;  % Patch size
Par.step        =   5; % gap between neighboorhood patches
Par.n_patch = 1200000; % the number of patches for training
Par.n_cls = 100; % the number of clusters
Par.n_img = min(100, im_num); % the number of noisy images for training

[XN, XC, XN0, XC0, Par] = rnd_smp_patch(TT_Original_image_dir, GT_Original_image_dir, Par);
XC = XC - repmat(mean(XC), [Par.ps2ch 1]);
XN = XN - repmat(mean(XN), [Par.ps2ch 1]);
XC0 = XC0 - repmat(mean(XC0), [Par.ps2ch 1]);
XN0 = XN0 - repmat(mean(XN0), [Par.ps2ch 1]);

[model, cls_idx]  =  emgm(XC, Par.n_cls);
for c = 1 : Par.n_cls
    idx = find(cls_idx == c);
    if (length(idx) >  3000000)
        select_idx = randperm(length(idx));
        idx = idx(select_idx(1:3000000));
    end
    Xn{c} = XN(:, idx);
    Xc{c} = XC(:, idx);
end
clear XN XC;
GMM_model = ['GMM_' num2str(Par.ps2ch) '_' num2str(Par.n_patch) '_' num2str(Par.n_cls) '_' IMname '_' datestr(now, 30) '.mat'];
save(GMM_model, 'model', 'Xn','Xc','Par');