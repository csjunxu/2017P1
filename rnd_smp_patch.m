function [XN, XC, XN0, XC0, Par] = rnd_smp_patch(TrainingNoisy, TrainingClean, Par)

Nim_path = fullfile(TrainingNoisy,'*.JPG');
Cim_path = fullfile(TrainingClean,'*.JPG');

Nim_dir = dir(Nim_path);
Cim_dir = dir(Cim_path);

Nim_num = length(Nim_dir);


% noisy patches per image
n_img = min(Par.n_img, Nim_num);
% extract noisy patches
XN = [];
XC = [];
XN0 = [];
XC0 = [];
nper_img = floor(Par.n_patch/n_img);
for ii = 1:n_img
    Nim = im2double(imread(fullfile(TrainingNoisy, Nim_dir(ii).name)));
    Cim = im2double(imread(fullfile(TrainingClean, Cim_dir(1).name)));
    
    [h, w, ch] = size(Nim);
    if h >= 512
        randh = randi(h-512);
        Nim = Nim(randh+1:randh+512, :, :);
        Cim = Cim(randh+1:randh+512, :, :);
    end
    if w >= 512
        randw = randi(w-512);
        Nim = Nim(:, randw+1:randw+512, :);
        Cim = Cim(:, randw+1:randw+512, :);
    end
    [N, C, N0, C0, Par] = sample_paired_patches(Nim, Cim, nper_img, Par);
    XN = [XN, N];
    XC = [XC, C];
    XN0 = [XN0, N0];
    XC0 = [XC0, C0];
end

% final results
patch_path = ['rnd_smp_patch_' num2str(Par.ps2ch) '_' num2str(Par.n_patch) '_' datestr(now, 30) '.mat'];
save(patch_path, 'XN', 'XC', 'XN0', 'XC0');