function im_out = offlinep_bscdl_BID_full(IMin_y,model,Dict,par,param)
[h,w,ch] = size(IMin_y);
% Used for calculate CSNR
par.height = h;
par.width = w;
par.te_num = ch;

psf = fspecial('gaussian', par.win+2, 2.2);
XN = data2patch(IMin_y,  par);
% Initial
im_out = IMin_y;

AN = zeros(par.K, size(XN, 2));
AC = zeros(par.K, size(XN, 2));
for t = 1 : par.nInnerLoop
    if t == 1
        YC = data2patch(conv2(im_out, psf, 'same') - im_out, par);
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
    XC = data2patch(im_out,  par);
    XN = data2patch(IMin_y,  par);
    meanX = repmat(mean(XC), [par.win^2 1]);
    XN = XN - meanX;
    XC = XC - meanX;
    for i = 1 : par.cls_num
        idx_cluster   = find(cls_idx == i);
        length_idx = length(idx_cluster);
        start_idx = [1, length_idx];
        for j  = 1 : length(start_idx) - 1
            idx_temp = idx_cluster(start_idx(j):start_idx(j+1));
            Xc    = double(XC(:, idx_temp));
            Xn    = double(XN(:, idx_temp));
            Dc    = Dict.DC{i};
            Dn    = Dict.DN{i};
            Uc    = Dict.UC{i};
            Un    = Dict.UN{i};
            if (t == 1)
                % Pn    = Dict.PN{i}; % suffer dimension problem 08/15/2016;
                Pn    = repmat(mean(Dict.PN{i},2), size(Xn)); % expectation of noise; 08/15/2016;
                Alphan = mexLasso(Xn - Pn, Dn, param);
                Alphac = Uc \ Un * Alphan;
                Xc = Dc * Alphac; % Xc->Xn; 07/07/2016;  Xn->Xc; 07/22/2016;
            else
                Alphac = AC(:, idx_temp);
            end
            D = [Dn; par.sqrtmu * Un]; % Wn ->Un 07/22/2016;
            Pn = (Xn - Dn * Alphan) / (1 + par.nu); % 08/15/2016;
            Y = [Xn - Pn; par.sqrtmu * Uc * full(Alphac)];
            Alphan = mexLasso(Y, D,param);
            % Pn = (Xn - Dn * Alphan) / (1 + par.nu); % 08/15/2016;
            clear Y D;
            %% CVPR2012 SCDL case
            D = [Dc; par.sqrtmu * Uc];
            Y = [Xc; par.sqrtmu * Un * full(Alphan)];
            Alphac = full(mexLasso(Y, D,param));
            clear Y D;
%             %% ICCV2013 MML case
%             Alphac = Uc \ Un * Alphan;
            %% Reconstruction
            Xc = Dc * Alphac;
            XC(:, idx_temp) = Xc;
            AN(:, idx_temp) = Alphan;
            AC(:, idx_temp) = Alphac;
        end
    end
      im_out = patch2data(XC+meanX, h, w, 1,par.win, par.step);
end