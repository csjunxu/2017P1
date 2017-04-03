% Main Function of Robust Coupled Dictionary Learning
% Input:
% Ax, Ay : Initial sparse coefficient of two domains
% X, Y     : Image Data Pairs of two domains
% Dx, Dy : Initial Dictionaries
% M        : Initial Projection Matrix
% Par      : Parameters
%
% Output
% Ax, Ay : Output sparse coefficient of two domains
% Dx, Dy : Output Coupled Dictionaries
% M        : Output Projection Matrix for Ax and Ay
%

function     [Dx, Dy, Ax, Ay, M, IniPSNR, PSNR] = RCDL_MAD2(X, Y, Dx, Dy, Ax, Ay, M, Par)

f = 0;
IniPSNR = csnr( Y * 255, X * 255, 0, 0 );
fprintf('Initial PSNR = %2.4f.\n', IniPSNR);

%% Iteratively solve D A U

for t = 1 : Par.nIter
    f_prev = f;
    if Par.IniType == 'Ran'
        %% Updating Dx and Dy
        [Ux, Sx, Vx] = svd(Ax * X', 'econ');
        Dx = Vx * Ux';
        [Uy, Sy, Vy] = svd(Ay * Y', 'econ');
        Dy = Vy * Uy';
    end
    
    %% Updating Ax and Ay
    Bx = (Par.alphax * Dx' * X + Par.beta * M * Ay) ./ (Par.alphax + Par.beta);
    threx = 0.5 * Par.gammax / (Par.alphax + Par.beta);
    Ax = sign(Bx).*max(abs(Bx) - threx, 0);
    
    By = (Par.alphay * Dy' * Y + Par.beta * M' * Ax) ./ (Par.alphay + Par.beta);
    threy = 0.5 * Par.gammay / (Par.alphay + Par.beta);
    Ay = sign(By).*max(abs(By) - threy, 0);
    
    %% Updating M
    [U, S, V] = svd(Ay * Ax', 'econ');
    M = V * U';
    
    %% calculate the  energy
    P1x = Par.alphax * norm(X - Dx * Ax, 'fro')^2;
    P1y = Par.alphay * norm(Y - Dy * Ay, 'fro')^2;
    P2   = Par.beta * norm(Ax - M * Ay, 'fro')^2;
    P3x = Par.gammax * norm(Ax, 1);
    P3y = Par.gammay * norm(Ay, 1);
    f = P1x + P1y + P2 + P3x + P3y;
    
    %% if converge then break
    Xhat = Dy * Ay;
    PSNR = csnr( Xhat * 255, X * 255, 0, 0 );
    fprintf('Iter: %d, Energy: %d, PSNR = %2.4f.\n', t, f, PSNR);
    if (abs(f_prev - f) / f < Par.epsilon) || (f_prev - f < 0 && t > 1)
        break;
    end
end

