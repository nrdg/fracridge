function [coef,alphas] = fracridge(X,fracs,y,tol,mode)

% function [coef,alphas] = fracridge(X,fracs,y,tol,mode)
%
% <X> is the design matrix (n x p) with different data points
%   in the rows and different regressors in the columns
% <fracs> is a vector (1 x f) of one or more fractions between 0 and 1.
%   Fractions can be exactly 0 or exactly 1. However, values in between
%   0 and 1 should be no less than 0.001 and no greater than 0.999.
%   For example, <fracs> could be 0:.05:1 or 0:.1:1.
% <y> is the data (n x b) with one or more target variables in the columns
% <tol> (optional) is a tolerance value such that eigenvalues
%   below the tolerance are treated as 0. Default: 1e-6.
% <mode> (optional) can be:
%   0 means the default behavior
%   1 means to interpret <fracs> as exact alpha values that are desired.
%     This case involves using the rotation trick to speed up the ridge
%     regression but does not involve the fraction-based tailoring method.
%     In the case that <mode> is 1, the output <alphas> is returned as [].
%   Default: 0.
%
% return:
%  <coef> as the estimated regression weights (p x f x b)
%    for all fractional levels for all target variables.
%  <alphas> as the alpha values (f x b) that correspond to the
%    requested fractions. Note that alpha values can be Inf
%    in the case that the requested fraction is 0, and can
%    be 0 in the case that the requested fraction is 1.
%
% The basic idea is that we want ridge-regression solutions
% whose vector lengths are controlled by the user. The vector
% lengths are specified in terms of fractions of the length of
% the full-length solution. The full-length solution is either
% the ordinary least squares (OLS) solution in the case of
% full-rank design matrices, or the pseudoinverse solution in
% the case of rank-deficient design matrices.
%
% Framing the problem in this way provides several benefits:
% (1) we don't have to spend time figuring out appropriate
% alphas for each regression problem, (2) when implemented
% appropriately, we can compute the full set of solutions
% very efficiently and in a way that avoids the need
% to compute X'*X (which might be large), and (3) parameterizing
% the ridge-regression problem in terms of fractional lengths
% provides nicely interpretable results.
%
% The fraction-based method is tailored to each target variable
% in the sense that different target variables may need different
% alpha values in order to achieve the same fractional length.
% The output <alpha> is provided in case the user wants to know
% exactly what alpha values were used for each target variable.
%
% Notes:
% - We silently ignore regressors that are all zeros.
% - The class of <coef> and <alphas> is matched to the class of <X>.
% - It is assumed that all values in <X> and <y> are finite.
% - If a given target variable consists of all zeros, this is
%   a degenerate case, and we will return regression weights that
%   are all zeros and alpha values that are all zeros.
%
% % Example 1:
%
% y = randn(100,1);
% X = randn(100,10);
% coef = inv(X'*X)*X'*y;
% [coef2,alpha] = fracridge(X,0.3,y);
% coef3 = inv(X'*X + alpha*eye(size(X,2)))*X'*y;
% coef4 = fracridge(X,alpha,y,[],1);
% norm(coef)
% norm(coef2)
% norm(coef2) ./ norm(coef)
% norm(coef2-coef3)
% norm(coef4-coef3)
%
% % Example 2:
%
% y = randn(1000,300);        % 1000 data points x 300 target variables
% X = randn(1000,3000);       % 1000 data points x 3000 predictors
%   % naive approach
% tic;
% alphas = 10.^(-4:.5:5.5);  % guess 20 alphas
% cache1 = X'*X;
% cache2 = X'*y;
% coef = zeros(3000,length(alphas),300);
% for j=1:length(alphas)
%   coef(:,j,:) = permute(inv(cache1 + alphas(j)*eye(size(X,2)))*cache2,[1 3 2]);
% end
% toc;
%   % fracridge approach
% tic;
% fracs = .05:.05:1;         % get 20 equally-spaced lengths
% coef2 = fracridge(X,fracs,y);
% toc;
%   % fracridge implementation of simple rotation
% tic;
% coef3 = fracridge(X,alphas,y,[],1);
% toc;
% assert(all(abs(coef(:)-coef3(:))<1e-4));

%% %%%%%% SETUP

% deal with inputs
if ~exist('tol','var') || isempty(tol)
  tol = 1e-6;
end
if ~exist('mode','var') || isempty(mode)
  mode = 0;
end

% internal constants
bbig   = 10^3;    % sets huge-bias alpha. this will reduce length to less than 1/(1+10^3) = 0.000999 of the full length.
bsmall = 10^-3;   % sets small-bias alpha. this will preserve length to at least 1/(1+10^-3) = 0.999 of the full length.
bstep  = 0.2;     % step size for alpha in log10 space
debugmode = 0;    % if set to 1, we do some optional sanity checks

% ignore bad regressors (those that are all zeros)
bad = all(X==0,1);
if any(bad)
  fprintf('WARNING: bad regressors detected (all zero). ignoring these regressors.\n');
  X = X(:,~bad);
end

% calc
n = size(X,1);      % number of data points
p = size(X,2);      % number of parameters (regressors)
b = size(y,2);      % number of distinct outputs (targets) being modeled
f = length(fracs);  % number of fractions (or alphas) being requested

%% %%%%%% PERFORM SVD AND ROTATE DATA

% decompose X [this is a costly step]
  % [u,s,v] = svd(X,'econ');  % when n>=p, u is n x p, s is p x p, v is p x p
  %                           % when n< p, u is n x n, s is n x n, v is p x n
if n >= p

  % avoid making a large u
  [~,s,v] = svd(X'*X,'econ');

  % extract the eigenvalues
  selt = sqrt(diag(s));  % p x 1
  clear s;               % clean up to save memory

  % rotate the data, i.e. u'*y (X=u*s*v', u=X*v*inv(s), u'=inv(s)*v'*X')
  % in the below, the operation is (pxp) x (pxp) x (pxn) x (nxb).
  % we group appropriately to speed things up.
  if b >= n
    ynew = (diag(1./selt) * v' * X') * y;  % p x b
  else
    ynew = diag(1./selt) * v' * (X' * y);  % p x b
  end

else

  % do it
  [u,s,v] = svd(X,'econ');

  % extract the eigenvalues
  selt = diag(s);  % n x 1
  clear s;         % clean up to save memory

  % rotate the data
  ynew = u'*y;     % n x b
  clear u;         % clean up to save memory

end

% calc
sz = length(selt);  % the size (rank) of the problem (either p or n)

% mark eigenvalues that are essentially zero
isbad = selt < tol;      % p x 1 (OR n x 1)
anyisbad = any(isbad);
if anyisbad
  fprintf('WARNING: some eigenvalues are being treated as 0.\n');
end

%% %%%%%% COMPUTE OLS SOLUTION IN ROTATED SPACE

% compute the OLS (or pseudoinverse) solution in the rotated space
ynew = ynew ./ repmat(selt,[1 b]);  % p x b (OR n x b)
if anyisbad
  ynew(isbad,:) = 0;  % the solution will be 0 along directions associated with eigenvalues that are essentially zero
end

%% %%%%%% DO THE MAIN STUFF

% initialize
if n >= p
  coef = zeros(p,f*b,class(X));  % this is the easy case. the final rotation doesn't change the dimensionality.
else
  coef = zeros(n,f*b,class(X));  % in this case, the final rotation will change from n dimensions to p dimensions.
end

% we have two modes of operation...
switch mode

% this is the case of fractions being requested
case 0

  %% %%%%% DO SOME SETUP

  % figure out a reasonable grid for alpha at reasonable level of granularity
  val1 = bbig*selt(1)^2;              % huge bias (take the biggest eigenvalue down massively)
  val2 = bsmall*min(selt(~isbad))^2;  % tiny bias (just add a small amount to the smallest eigenvalue)
  alphagrid = fliplr([0 10.^(floor(log10(val2)):bstep:ceil(log10(val1)))]);  % huge bias to tiny bias to no bias (1 x g)
  g = length(alphagrid);

  % note that alphagrid is like [10e6 ... 0].
  %
  % also, note that the <bstep> is a heuristic. we just need to sample alphas at sufficient
  % granularity such that linear interpolation will be able to find the requested
  % fractions with sufficient accuracy. the idea here is that the computations for
  % constructing solutions with different regularization amounts (see below) are cheap,
  % and so we are happy to "oversample" to some extent.

  % construct scaling factor
  seltSQ = selt.^2;                                                     % p x 1 (OR n x 1)
  scLG = repmat(seltSQ,[1 g]);                                          % p x g (OR n x g)
  scLG = scLG ./ (scLG + repmat(alphagrid,[size(scLG,1) 1]));
  if anyisbad
    scLG(isbad,:) = 0;                                                  % for safety, ensure bad eigenvalues get scalings of 0
  end

  % pre-compute for speed
  scLG = scLG.^2';                                                      % g x p (OR g x n)
  seltSQ2 = repmat(seltSQ,[1 f]);                                       % p x f (OR n x f)
  fracisz = find(fracs==0);                                             % indices into <fracs>
  logalpha = log(1+alphagrid)';                                         % transform alphas to log(1+x) scale (g x 1)

  % init
  alphas = zeros(f,b,class(X));

  %% %%%%% PROCEED TO COSTLY FOR-LOOP

  % compute ridge regression solutions and corresponding alphas
  for ii=1:b

    % compute vector length for each alpha in the grid
    len = sqrt(scLG*ynew(:,ii).^2);  % g x 1

    % when ynew is all 0, this is a degenerate case
    % and will cause len to be a bunch of zeros.
    % let's check the last element (no regularization),
    % and if it's 0, we know this is the degenerate case.
    % if so, we just continue, and this will result in alphas
    % and coeff to be left at 0.
    if len(end)==0
      continue;
    end

    % express lengths as fractions relative to the full length (i.e. no regularization)
    len = len / len(end);

    % inspection (is the proposed interpolation scheme reasonable?)
    if debugmode
      figure;
      scatter(len,logalpha,'ro');
      xlabel('length');
      ylabel('log(1+alphagrid)');
    end

    % sanity check that the gridding is not too coarse
    mxgap = max(abs(diff(len)));                    % maximum gap
    assert(mxgap < 0.2,'need to decrease bstep!');  % if this fails, bstep should be smaller

    % use linear interpolation to determine alphas that achieve the desired fractional levels.
    % we interpolate in log(1+x) space in order to help achieve good quality interpolation.
    temp = interp1qr(len,logalpha,fracs');          % f x 1 (if out of range, will be NaN; we check this later)
    temp = exp(temp)-1;                             % undo the log transform
    temp(fracisz) = Inf;                            % when frac is exactly 0, we are out of range, so handle explicitly
    alphas(:,ii) = temp;

    % apply scaling to the OLS solution
    coef(:,(ii-1)*f+(1:f)) = repmat(seltSQ .* ynew(:,ii),[1 f]) ./ (seltSQ2 + repmat(temp',[sz 1]));

  end

  % accuracy check to see if the achieved vector lengths are close to what was requested
  if debugmode
    temp = sqrt(sum(coef.^2,1));    % 1 x f*b
    temp2 = sqrt(sum(ynew.^2,1));   % 1 x b
    temp3 = reshape(temp,[f b]) ./ repmat(temp2,[f 1]);  % f x b
    temp4 = repmat(fracs',[1 b]);
    figure; hold on;
    scatter(temp3(:),temp4(:),'r.');
    xlabel('empirical fraction');
    ylabel('requested fraction');
    plot([0 1],[0 1],'g-');
  end

  % for safety, ensure bad eigenvalues get scalings of 0
  if anyisbad
    coef(isbad,:) = 0;
  end

  % if all went well, no value should be NaN (but can be Inf)
  assert(~any(isnan(alphas(:))),'NaN encountered in alphas. Is an element in <fracs> too close to 0?');

  % rotate solution to the original space
  coef = reshape(v*coef,[p f b]);

% this is the case of conventional alphas being requested
case 1

  % construct scaling factor
  sc = repmat(selt.^2,[1 f]);                                 % p x f (OR n x f)
  sc = sc ./ (sc + repmat(fracs,[size(sc,1) 1]));             % p x f (OR n x f)
  if anyisbad
    sc(isbad,:) = 0;                                          % for safety, ensure bad eigenvalues get scalings of 0
  end

  % apply scaling to the OLS solutions.
  % do it in a for-loop to save memory usage.
  for ii=1:b
    coef(:,(ii-1)*f+(1:f)) = sc .* repmat(ynew(:,ii),[1 f]);
  end

  % rotate solution to the original space
  coef = reshape(v*coef,[p f b]);

  % deal with output (alphas is irrelevant, so set to [])
  alphas = cast([],class(X));

end
