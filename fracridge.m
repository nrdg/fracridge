function [h,lambdas] = ridgeregressiongamma(X,gamma,y,tol)

% function [h,lambdas] = ridgeregressiongamma(X,gamma,y,tol)
%
% <X> is the design matrix (n x p) with data points in the rows
%   and regressors in the columns
% <gamma> is a vector of one or more fractions between 0-1 (1 x f)
% <y> is the data (n x b) with one or more outcome variables in the columns
% <tol> (optional) is a tolerance value such that eigenvalues
%   below the tolerance are treated as 0. Default: 1e-6.
%
% return:
%  <h> as the estimated regression weights (p x f x b)
%    for all fractional levels for all outcome variables.
%  <lambdas> as the lambda values that correspond to the
%    requested fractions (f x b). Note that <lambdas> can
%    be Inf in the case that the requested fraction is 0.
%
% The basic idea is that we want ridge-regression solutions
% whose vector lengths are controlled by the user. The vector
% lengths are specified in terms of fractions of the length of 
% the full-length solution which is the ordinary least squares
% (OLS) solution. Framing the problem in this way provides 
% several benefits: (1) we don't have to spend time figuring out
% appropriate lambdas for each regression problem, (2) when 
% implemented appropriately, we can compute the full set of 
% solutions very efficiently and in a way that avoids the need 
% to compute X'*X (which might be large). Computational
% benefits are likely to occur for large-scale problems
% where p is large. For small problems, we may take a hit in 
% computational time due to various overhead-related steps,
% but this is probably not a big deal since the problem is
% fast to solve anyway.
%
% Notes:
% - We silently ignore regressors that are all zeros.
% - The class of <h> and <lambdas> is matched to the class of <X>.
%
% Example 1:
%
% y = randn(100,1);
% X = randn(100,10);
% h = inv(X'*X)*X'*y;
% [h2,lambda] = ridgeregressiongamma(X,0.3,y);
% h3 = inv(X'*X + lambda*eye(size(X,2)))*X'*y;
% norm(h)
% norm(h2)
% norm(h2) ./ norm(h)
% norm(h2-h3)
%
% Example 2:
%
% y = randn(1000,300);        % 1000 data points x 300 outcome variables
% X = randn(1000,3000);       % 1000 data points x 3000 predictors
%   % naive approach
% tic;
% lambdas = 10.^(-4:.5:5.5);  % guess 20 lambdas
% cache1 = X'*X;
% cache2 = X'*y;
% h = zeros(3000,300,length(lambdas));
% for j=1:length(lambdas)
%   h(:,:,j) = inv(cache1 + lambdas(j)*eye(size(X,2)))*cache2;
% end
% toc;
%   % ridgeregressiongamma's approach
% tic;
% gammas = .05:.05:1;         % get 20 equally-spaced vector lengths
% h2 = ridgeregressiongamma(X,gammas,y);
% toc;

% deal with inputs
if ~exist('tol','var') || isempty(tol)
  tol = 1e-6;
end

% internal constants
bbig   = 10^3;    % sets huge-bias
bsmall = 10^-3;   % sets small-bias
bstep  = 0.25;    % step size in log10 space

% ignore bad regressors (those that are all zeros)
bad = all(X==0,1);
if any(bad)
  X = X(:,~bad);
end

% calc
n = size(X,1);     % number of data points
p = size(X,2);     % number of parameters (regressors)
b = size(y,2);     % number of distinct outputs being modeled
f = length(gamma); % number of fractions being requested

% decompose X [this is a costly step]
    % [u,s,v] = svd(X,'econ');  % when n>=p, u is n x p, s is p x p, v is p x p
    %                           % when n< p  u is n x n, s is n x n, v is p x n
if n >= p

  % do it but avoid making a large u
  [~,s,v] = svd(X'*X,'econ');

  % extract the eigenvalues
  selt = sqrt(diag(s));  % p x 1
  clear s;         % clean up

  % rotate the data (X=u*s*v', u=X*v*inv(s), u'=inv(s)*v'*X')
  if b >= n
    ynew = (diag(1./selt) * v' * X') * y;  % p x b  (notice grouping to speed things up)
  else
    ynew = diag(1./selt) * v' * (X' * y);  % p x b
  end

else

  % do it
  [u,s,v] = svd(X,'econ');

  % extract the eigenvalues
  selt = diag(s);  % n x 1
  clear s;         % clean up

  % rotate the data
  ynew = u'*y;  % n x b
  clear u;      % clean up

end

% mark eigenvalues that are essentially zero
isbad = selt < tol;
anyisbad = any(isbad);
if anyisbad
  fprintf('WARNING: some eigenvalues are being treated as 0.\n');
end

% compute the OLS solution in the rotated space
ynew = ynew ./ repmat(selt,[1 size(ynew,2)]);  % p x b (OR n x b)
if anyisbad
  ynew(isbad,:) = 0;  % the solution will be 0 for eigenvalues that are essentially zero (pseudoinverse)
end

% figure out a reasonable range for lambda at reasonable level of granularity
val1 = bbig*selt(1)^2;              % huge bias (take the biggest eigenvalue down to ~.001 of what OLS would be)
val2 = bsmall*min(selt(~isbad))^2;  % tiny bias (just add a small amount to the smallest eigenvalue)
lambdagrid = [0 10.^(floor(log10(val2)):bstep:ceil(log10(val1)))];  % no bias to tiny bias to huge bias (1 x lambagrid)
fprintf('using a lambagrid of size %d.\n',length(lambdagrid));

% note that the <bstep> is a heuristic. we just need to sample lambdas at sufficient
% granularity such that pchip interpolation will be able to find the requested
% fractions with sufficient accuracy. the idea here is that the computations for 
% constructing solutions with different regularization amounts (see below) are cheap,
% and so we are happy to "oversample" to some extent.

% pre-compute for speed
seltSQ = selt.^2;

% pre-compute for speed
scLG = repmat(seltSQ,[1 length(lambdagrid)]);
scLG = scLG ./ (scLG + repmat(lambdagrid,[length(seltSQ) 1]));         % p x lambdagrid (OR n x lambdagrid)
if anyisbad
  scLG(isbad,:) = 0;                                                     % for safety, ensure bad eigenvalues get scalings of 0 
end
scLG = scLG.^2';                                                       % lambdagrid x p (OR lambdagrid x n)
seltSQ2 = repmat(seltSQ,[1 f]);                                        % p x f (OR n x f)

% compute the ridge regression solutions and the corresponding lambdas
if n >= p
  h = zeros(p,f*b,class(X));
else
  h = zeros(n,f*b,class(X));
end
lambdas = zeros(f,b,class(X));
for ii=1:b

  % compute vector length for each lambda
    % OLD SLOW WAY: len = vectorlength(bsxfun(@times,scLG,ynew(:,ii)),1);   % 1 x lambdagrid
  len = sqrt((scLG*ynew(:,ii).^2)');                     % 1 x lambdagrid
  
  % make lengths relative to the first one (i.e. no regularization)
  len = len / len(1);

  % inspection (is the proposed interpolation scheme accurate?)
  if 0
    figure;
    scatter(len,log(1+lambdagrid),'ro');
    xlabel('length');
    ylabel('log(1+lambdagrid)');
  end
  
  % sanity check that the gridding is not too coarse
  mxgap = max(abs(diff(len)));                    % maximum gap
  assert(mxgap < 0.2,'need to decrease bstep!');  % if this fails, bstep should be smaller
  assert(min(len) < 0.001);                       % if all is well, we should have sampled close to 0

  % use cubic interpolation to find lambdas that achieve the desired fractional levels.
  % we interpolate in log(1+y) space in order to help achieve nice smooth curves.
  temp = interp1(len,log(1+lambdagrid),gamma,'pchip',NaN);             % 1 x f
  temp = exp(temp)-1;                                                  % undo the log transform
  temp(gamma==0) = Inf;                                                % when gamma is exactly 0, we are out of range, so handle explicitly
  assert(all(~isnan(temp)));                                           % if all went well, no value should be NaN (but can be Inf)
  lambdas(:,ii) = temp;
  
  % get the solutions
  sc = seltSQ2 ./ (seltSQ2 + repmat(temp,[length(seltSQ) 1]));             % p x f (OR n x f)
    %OLD: sc = bsxfun(@rdivide,seltSQ,bsxfun(@plus,seltSQ,temp));              % p x f (OR n x f)
  if anyisbad
    sc(isbad,:) = 0;                                                     % for safety, ensure bad eigenvalues get scalings of 0
  end
    %h(:,:,ii) = v*(sc.*repmat(ynew(:,ii),[1 size(sc,2)]));  % rotate solution to original space
  h(:,(ii-1)*f+(1:f)) = sc.*repmat(ynew(:,ii),[1 f]);

end

% rotate solution to original space
h = reshape(v*h,[p f b]);
