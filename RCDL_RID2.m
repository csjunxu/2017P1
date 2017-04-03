function [rI, PN] = RCDL_RID2(nI, I, model, RCDL, Par)
[h, w, ch] = size(nI);
% Used for calculate PSNR and SSIM 
Par.height = h;
Par.width = w;
Par.te_num = ch;

psf = fspecial('gaussian', Par.win+2, 2.2);
XN = data2patch(nI,  Par);
% Initial
rI = nI;
PN = cell(1,Par.cls_num);
AN = zeros(Par.K, size(XN, 2));
AC = zeros(Par.K, size(XN, 2));
for t = 1 : Par.nInnerLoop
    if t == 1
        YC = im2patch(conv2(rI, psf, 'same') - rI, Par);
        %% GMM: full posterior calculation
        PYZ = zeros(model.nmodels,size(YC,2));
        for i = 1:model.nmodels
            sigma = model.covs(:,:,i);
            [R,~] = chol(sigma);
            Q = R'\YC;
            PYZ(i,:)  = - sum(log(diag(R))) - dot(Q,Q,1)/2;
        end
        %% find the most likely component for each patch group
        [~,cls_idx] = max(PYZ);
    end
    XC = im2patch(rI,  Par);
    XN = im2patch(nI,  Par); % one time is ok
    meanX = repmat(mean(XC), [Par.win^2 1]);
    XN = XN - meanX;
    XC = XC - meanX;
    for i = 1 : Par.cls_num
        idx_cluster   = find(cls_idx == i);
        length_idx = length(idx_cluster);
        start_idx = [1, length_idx];
        idx_temp = idx_cluster(start_idx(1):start_idx(2));
        X    = double(XC(:, idx_temp));
        Y    = double(XN(:, idx_temp));
        D    = RCDL.D{i};
        M    = RCDL.M{i};
        if (t == 1)
            Pn    = zeros(size(Y));
            Ay = mexLasso(Y - Pn, Dn, param);
            Ax = M * Ay;
            X = D * Ax;
        else
            Pn = PN{i};
            Ax = AC(:, idx_temp);
        end  
        D = [Dn; Par.sqrtmu * Un]; 
        Y = [Y - Pn; Par.sqrtmu * M * full(Ax)];
        Ay = mexLasso(Y, D,param);
        clear Y D;
        %% CVPR2012 SCDL case
        D = [D; Par.sqrtmu * M];
        Y = [X; Par.sqrtmu * Un * full(Ay)];
        Ax = full(mexLasso(Y, D,param));
        clear Y D;
        %% Reconstruction
        X = D * Ax;
        XC(:, idx_temp) = X;
        AN(:, idx_temp) = Ay;
        AC(:, idx_temp) = Ax;
    end
    rI = patch2data(XC+meanX, h, w, 1,Par.win, Par.step);
    fprintf(['Loop: ' num2str(t) ', PSNR =' num2str(csnr( rI*255, I*255, 0, 0 )) ', SSIM =' num2str(cal_ssim( rI*255, I*255, 0, 0  )) '. \n' ]);
end