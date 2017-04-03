clear;

load GMM_108_1000000_100_Canon_80D_ISO800_20170329T180641.mat;

Par.epsilon         =        1e-3;
Par.IniType = 'Ran';
% SVD
% Ran

for alphay =  [1]
    Par.alphay = alphay;
    for alphax = [100]
        Par.alphax = alphax;
        Par.sqrtalphax = sqrt(Par.alphax);
        Par.sqrtalphay = sqrt(Par.alphay);
        for beta =  [100]
            Par.beta = beta;
            for gammax = [1 0.5 0.1 0.05 0.01 0.005 0.001]
                Par.gammax = gammax;
                for gammay = [1 0.5 0.1 0.05 0.01 0.005 0.001]
                    Par.gammay = gammay;
                    % Initiatal model
                    RCDL = [];
                    IniPSNR = zeros(Par.n_cls,1);
                    PSNR = zeros(Par.n_cls,1);
                    SSIM = zeros(Par.n_cls,1);
                    for i = 1 : Par.n_cls
                        Par.cls = i;
                        X = double(Xc{i});
                        Y = double(Xn{i});
                        X = X - repmat(mean(X), [Par.ps2ch 1]);
                        Y = Y - repmat(mean(Y), [Par.ps2ch 1]);
                        if Par.IniType == 'Ran'
                            %% random initialization
                            Dx = orth(randn(size(X, 1), size(X, 1)));
                            Dy = orth(randn(size(Y, 1), size(Y, 1)));
                            Par.nIter  = 500;
                        elseif Par.IniType == 'SVD'
                            %% Initialized by SVD of X
                            [Ux, Sx, Vx] = svd(X, 'econ');
                            [Uy, Sy, Vy] = svd(Y, 'econ');
                            Dx = Ux;
                            Dy = Uy;
                            Par.nIter           =       1;
                        end
                        
                        Ax = Dx' * X;
                        Ay = Dy' * Y;
                        [U, S, V] = svd(Ax * Ay', 'econ');
                        M = U * V';
                        %                         RCDL_RID_MAD_Initial = sprintf('RCDL_Initial_MAD.mat');
                        %                         save(RCDL_RID_MAD_Initial,'Dx', 'Dy', 'Ax', 'Ax', 'M');
                        fprintf('Robustly coupled dictionary learning: Cluster: %d\n', i);
                        [Dx, Dy, Ax, Ay, M, IniPSNRi, PSNRi] = RCDL_MAD2(X, Y, Dx, Dy, Ax, Ay, M, Par);
                        RCDL.Dx{i} = Dx;
                        RCDL.Dy{i} = Dy;
                        RCDL.M{i} = M;
                        IniPSNR(i) = IniPSNRi;
                        PSNR(i) = PSNRi;
                        %                         RCDL_RID_MAD = sprintf('RCDL_RID_MAD1_ax%2.3f_ay%2.3f_b%2.3f_gx%2.3f_gy%2.3f.mat', alphax, alphay, beta, gammax, gammay);
                        %                         save(RCDL_RID_MAD, 'RCDL', 'PSNR', 'SSIM');
                    end
                    mIniPSNR = mean(IniPSNR);
                    mPSNR = mean(PSNR);
                    RCDL_RID_MAD = sprintf(['RCDL_RID_MAD2_' Par.IniType '_ax' num2str(alphax) '_ay' num2str(alphay) '_b' num2str(beta) '_gx' num2str(gammax) '_gy' num2str(gammay) '_' datestr(now, 30) '.mat']);
                    save(RCDL_RID_MAD,'RCDL', 'IniPSNR', 'mIniPSNR', 'PSNR', 'mPSNR');
                end
            end
        end
    end
end
