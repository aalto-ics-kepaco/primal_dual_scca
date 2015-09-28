function [wa, e, corval, corval_without_bestseed, resval,kout, W, Z]...
    = SCCAwrapper_cvx(trainX,Kb,k,sk,mode)

% This is a wrapper function that runs the scca_cvx_singleprog_tau function
% with a greedy (maximum correlation) choice of e_k in each deflation stage
%
% Input:
%   trainX - first view train data
%   Kb     - second view train data
%   k      - the medoids from the second view obtained as an output from
%            the spectral_clustering function
%   sk     - the hyperparameter controlling the number of zero entries of 
%            the weight vector wa (eg sk=0.1 few zeros, sk=1.5 many zeros)
%   mode   - 0: use vector k for e_k's, flatten, 1: find optimal e_k for each
%            deflation, flatten, 2: don't flatten
%
% Output:
%   wa     - weights for the primal view
%   e      - dual weights for the kernelized view
%   corval - canonical correlation coefficients for the projections
%   cor..w.- canonical correlation coefficients if the best seed is
%            excluded
%   W      - Projection for primal
%   Z      - Projection for dual
%   output - a struct with various values

% David R. Hardoon 25/06/2007 
% http://homepage.mac.com/davidrh/
% D.Hardoon@cs.ucl.ac.uk
%
% Modified by Juho Rousu from the original code written by David R. Hardoon
% Modified by Nicolas Hoyo from the original code written by David R. Hardoon
% Modified by Anna Cichonska
% Modified by Viivi Uurtio

a=1; b=0;
tX = trainX;
KK = Kb;
co = 1;
%l = size(KK,1);
for i=1:length(k)
    %fprintf('\n Component %d\n',i);
    if mode==1
        % look for optimal e_k
        max_cor  = -1;
        for h=1:length(k)
            disp(['projection ' num2str(h)])
            
            [output.w,output.e,output.cor,output.res] =...
                scca_cvx_singleprog_tau(tX,KK,k(h),sk); %,corr_mode
                 
             % Values less than threshold are put to zero (CVX)
             output.w(abs(output.w)<1e-6)=0; 
             output.e(abs(output.e)<1e-6)=0;
             
             vec1=output.w'*tX; vec2=KK*output.e;
             vec1(k(h))=[]; vec2(k(h),:)=[];
             if norm(vec1)*norm(vec2)~=0
                out_corval_without_bestseed=vec1*vec2/(norm(vec1)*norm(vec2));
             else
                 out_corval_without_bestseed=0;
             end
            
            if out_corval_without_bestseed > max_cor && length(find(output.e)) > 0                                                 
                max_cor = out_corval_without_bestseed;
                kout(co) = k(h);
                wa(:,co) = output.w;
                e(:,co) = output.e;
                resval(co) = output.res;
                corval(co) = output.cor;                  
                corval_without_bestseed(co)=out_corval_without_bestseed;
                disp(['CAN CORR: ' num2str(corval_without_bestseed(co))])
                disp(['Seed: ' num2str(k(h))])
            end
        end
        if size(e,2) < co
                break
        end
    else
        [output.w,output.e,output.cor,output.res] =...
            scca_cvx_singleprog_tau(tX,KK,k(i),sk);%out_converged

          % Values less than threshold are put to zero (CVX) 
          output.w(abs(output.w)<1e-6)=0; 
          output.e(abs(output.e)<1e-6)=0;
          
          vec1=output.w'*tX; vec2=KK*output.e;
          vec1(k(i))=[]; vec2(k(i),:)=[];
          corval_without_bestseed(co)=vec1*vec2/(norm(vec1)*norm(vec2));
        kout(co) = k(i);
        wa(:,co) = output.w;
        e(:,co) = output.e;
        resval(co) = output.res;
        corval(co) = output.cor;
    end
    co = co + 1;
    
    if mode~=2
        % Dual Deflation
        projk(:,i) = KK*e(:,i);
        tau(:, i) = KK*projk(:,i);
        P = eye(length(KK)) - (tau(:,i)*tau(:,i)')/(tau(:,i)'*tau(:,i));
        KK = P'*KK*P;
        D= diag(sqrt(1./diag(KK)));
        KK= D*KK*D;
        
        % Primal Deflation
        proj(:,i) = tX*(tX'*wa(:,i));
        t(:,i) = tX'*proj(:,i);
        tX = tX - tX*(t(:,i)*t(:,i)')/(t(:,i)'*t(:,i));
    end
end
disp(' ');


if mode~=2
    % Primal projection
    P = trainX*t*inv(t'*t);
    W = (proj*inv(P'*proj));
    %%
   
    W  = W./ (ones(size(W,1),1) * sqrt( ones(1,size(trainX',1)) * ( (trainX'*W).^2 ) ) );
    
    % Dual Projection
    Z = projk*inv(inv(tau'*tau)*tau'*Kb*projk);
    Z  = Z./ (ones(size(Z,1),1) * sqrt( ones(1,size(Kb,1)) * ( (Kb*Z).^2 ) ) );
end

[corval_without_bestseed,Isorted] = sort(corval_without_bestseed,'descend');
wa = wa(:,Isorted);
e = e(:,Isorted);
resval = resval(Isorted);
kout = kout(:,Isorted); 
corval = corval(:,Isorted);

output = [];
output.kout = kout;
output.primal.w = wa;
output.dual.e = e;
%output.primal.P = P;
%output.dual.tau = tau;
%output.primal.tau = t;
output.cor = corval;
output.cor_corrected=corval_without_bestseed;
output.res = resval;
end