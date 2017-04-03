clear;

% GT_Original_image_dir = 'C:/Users/csjunxu/Desktop/CVPR2017/our_Results/Real_MeanImage/';
% GT_fpath = fullfile(GT_Original_image_dir, 'Canon_80D_ISO800*.JPG');
% TT_Original_image_dir = 'C:/Users/csjunxu/Desktop/CVPR2017/our_Results/Real_NoisyImage/';
% TT_fpath = fullfile(TT_Original_image_dir, 'Canon_80D_ISO800*.JPG');

% GT_Original_image_dir = 'C:/Users/csjunxu/Desktop/CVPR2017/cc_Results/Real_MeanImage/';
% GT_fpath = fullfile(GT_Original_image_dir, '*.png');
% TT_Original_image_dir = 'C:/Users/csjunxu/Desktop/CVPR2017/cc_Results/Real_NoisyImage/';
% TT_fpath = fullfile(TT_Original_image_dir, '*.png');

GT_Original_image_dir = 'C:/Users/csjunxu/Desktop/CVPR2017/cc_Results/Real_ccnoise_denoised_part/';
GT_fpath = fullfile(GT_Original_image_dir, '*mean.png');
TT_Original_image_dir = 'C:/Users/csjunxu/Desktop/CVPR2017/cc_Results/Real_ccnoise_denoised_part/';
TT_fpath = fullfile(TT_Original_image_dir, '*real.png');

GT_im_dir  = dir(GT_fpath);
TT_im_dir  = dir(TT_fpath);
im_num = length(TT_im_dir);

%% load parameters and dictionary
load RCDL_RID_MAD1_Ran_ax100_ay1_b100_gx0.001_gy0.001_20170331T230158.mat RCDL;
load GMM_108_1000000_100_Canon_80D_ISO800_20170329T180641.mat;

PSNR = zeros(1,im_num);
SSIM = zeros(1,im_num);

dataset = 'CC';

LG = 'Local';
% Local : Only use the noisy image and the mapping matrices
% Global : Use the noisy image, the latent clean image, and the mapping matrices

Par.Re = '1';
% 1 : Do not use the mapping matrices for reconstruction
% 2£ºUse the mapping matrices for reconstruction

Par.nInnerLoop = 4;

for alphay =  [1]
    Par.alphay = alphay;
    for alphax = [100]
        Par.alphax = alphax;
        for beta =  [100]
            Par.beta = beta;
            for gammax = [1]
                Par.gammax = gammax;
                for gammay =  [0.005:0.005:0.03]
                    Par.gammay = gammay;
                    for i = 1 : im_num
                        Par.ImInd = i;
                        I = im2double(imread(fullfile(GT_Original_image_dir, GT_im_dir(i).name)));
                        nI = im2double(imread(fullfile(TT_Original_image_dir, TT_im_dir(i).name)));
                        S = regexp(TT_im_dir(i).name, '\.', 'split');
                        IMname = S{1};
                        fprintf('%s : \n',IMname);
                        fprintf('The initial PSNR = %2.4f, SSIM = %2.4f. \n',csnr( nI*255, I*255, 0, 0 ),cal_ssim( nI*255, I*255, 0, 0 ));
                        [h,w,ch] = size(nI);
                        %%
                        if strcmp(LG, 'Local') == 1
                            [rI, Par] = RCDL_RID1(nI, I, model, RCDL, Par);
                        elseif strcmp(LG, 'Global') == 1
                            [rI, Par] = RCDL_RID2(nI, I, model, RCDL, Par);
                        end
                        fprintf('The final PSNR = %2.4f, SSIM = %2.4f. \n', csnr( rI*255, I*255, 0, 0 ), cal_ssim( rI*255, I*255, 0, 0  ));
                        %% output image
                        %                         if strcmp(LG, 'Local') == 1
                        %                             imwrite(rI, ['RCDL_RID_' method '_' Par.Re '_' dataset num2str(im_num) '_' IMname '.png']);
                        %                         elseif strcmp(LG, 'Global') == 1
                        %                             imwrite(rI, ['RCDL_RID_' method '_' dataset num2str(im_num) '_' IMname '.png']);
                        %                         end
                    end
                    %% output the results
                    PSNR = Par.PSNR;
                    SSIM = Par.SSIM;
                    mPSNR = mean(Par.PSNR,2);
                    mSSIM = mean(Par.SSIM,2);
                    if strcmp(LG, 'Local') == 1
                        savename = ['RCDL_RID_' LG '_' Par.Re '_' dataset num2str(im_num) '_ay' num2str(alphay) '_gy' num2str(gammay) '.mat'];
                        save(savename, 'mPSNR', 'mSSIM', 'PSNR', 'SSIM');
                    elseif strcmp(LG, 'Global') == 1
                        savename = ['RCDL_RID_' LG '_' dataset num2str(im_num) '_ax' num2str(alphax) '_ay' num2str(alphay) '_b' num2str(beta) '_gx' num2str(gammax) '_gy' num2str(gammay) '.mat'];
                        save(savename, 'mPSNR', 'mSSIM', 'PSNR', 'SSIM');
                    end
                    
                end
            end
        end
    end
end