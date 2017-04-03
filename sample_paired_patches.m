function [N, C, N0, C0, Par] = sample_paired_patches(nim, cim, n_patch, Par)

[h, w, ch]  = size(nim);
Par.h = h;
Par.w = w;
Par.ch = ch;
Par.maxr = Par.h - Par.ps + 1;
Par.maxc = Par.w - Par.ps + 1;
r          =  1:Par.step:Par.maxr;
Par.r          =  [r r(end) + 1:Par.maxr];
c          =  1:Par.step:Par.maxc;
Par.c          =  [c c(end) + 1:Par.maxc];
Par.lenr = length(Par.r);
Par.lenc = length(Par.c);
Par.ps2 = Par.ps^2;
Par.ps2ch = Par.ps2 * Par.ch;
% Total number of patches in the test image
Par.maxrc = Par.maxr * Par.maxc;
% Total number of seed patches being processed
Par.lenrc = Par.lenr * Par.lenc;


npatch = Image2PatchNew( nim, Par );
cpatch = Image2PatchNew( cim, Par );

Tn_patch = size(npatch, 2);
% p = randperm(Tn_patch);
% patch_num = min(n_patch, Tn_patch);
% p = p(1:patch_num);

p = 1:floor(Tn_patch/n_patch):Tn_patch;


N = npatch(:, p);
C = cpatch(:, p);

% find the small variance patch
smooth = find(var(C)<0.001);
N0 = N(smooth);
C0 = C(smooth);


fprintf('We have sampled %d patches.\r\n', length(p));
end
