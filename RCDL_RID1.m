function [rI, Par]  = RCDL_RID1(nI, I, model, RCDL, Par)
[h, w, ch] = size(nI);
Par.h = h;
Par.w = w;
Par.ch = ch;
Par.maxr = Par.h-Par.ps+1;
Par.maxc = Par.w-Par.ps+1;
Par.maxrc = Par.maxr * Par.maxc;
r = 1:Par.step:Par.maxr;
Par.r = [r r(end)+1:Par.maxr];
c = 1:Par.step:Par.maxc;
Par.c = [c c(end)+1:Par.maxc];
Par.lenr = length(Par.r);
Par.lenc = length(Par.c);
Par.lenrc = Par.lenr * Par.lenc;
Par.ps2 = Par.ps^2;
Par.ps2ch = Par.ps2 * Par.ch;
% Initial rI as nI
rI = nI;
for t = 1 : Par.nInnerLoop
    
    %% extract patches from image
    Y = im2patch(rI,  Par);
    meanY = repmat(mean(Y), [Par.ps2ch 1]);
    Y = Y - meanY;
    X = zeros(Par.ps2ch, Par.lenrc);
    %% clustering
    if mod(t, 2) == 1
        % GMM: full posterior calculation
        PY = zeros(model.nmodels, Par.lenrc);
        for i = 1:model.nmodels
            sigma = model.covs(:,:,i);
            [R,~] = chol(sigma);
            Q = R'\Y;
            PY(i,:)  = - sum(log(diag(R))) - dot(Q,Q,1)/2;
        end
        % find the most likely component for each patch group
        [~,cls_idx] = max(PY);
        cls_idx = cls_idx';
        [idx_patch,  s_idx] = sort(cls_idx);
        idx2 = idx_patch(1:end-1) - idx_patch(2:end);
        seq = find(idx2);
        seg = [0; seq; length(cls_idx)];
    end
    %     meanX = repmat(mean(Y), [Par.ps2ch 1]);
    %     Y = Y - meanX;
    for   j = 1:length(seg)-1
        idx_patch =   s_idx(seg(j)+1:seg(j+1));
        idx_cluster =   cls_idx(idx_patch(1));
        D    = RCDL.D{idx_cluster};
        M    = RCDL.M{idx_cluster};
        B = D' * Y(:,idx_patch);
        lambda = 0.5 * Par.gammay/Par.alphay;
        Ay = sign(B) .* max(abs(B) - lambda, 0);
        if strcmp(Par.Re, '1') == 1
            %% Reconstruction 1
            X(:, idx_patch) = D * Ay;
        elseif strcmp(Par.Re, '2') == 1
            %% Reconstruction 2
            %             Ax = M * Ay;
            X(:, idx_patch) = D * M * Ay;
        end
    end
    rI = Patch2Im(X + meanY, Par);
    %% calculate the PSNR and SSIM
    Par.PSNR(t, Par.ImInd) =   csnr( rI * 255, I * 255, 0, 0 );
    Par.SSIM(t, Par.ImInd)      =  cal_ssim( rI * 255, I * 255, 0, 0 );
    fprintf('Iter %d : PSNR = %2.4f, SSIM = %2.4f\n', t, Par.PSNR(t, Par.ImInd), Par.SSIM(t, Par.ImInd));
end
rI(rI > 1) = 1;
rI(rI < 0) = 0;
return;
end



