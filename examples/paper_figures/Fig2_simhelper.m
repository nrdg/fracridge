function [etime,stime,X,y] = simhelper(n,p,b,flag,hyper,X,y)

% function [etime,stime,X,y] = simhelper(n,p,b,flag,hyper,X,y)
%
% <n> is number of data points
% <p> is number of predictors
% <b> is number of targets
% <flag> is 1 (standard) or 2 (frac) or 3 (fracALT)
% <hyper> is vector of hyperparameters
%
% return:
% <etime> is time elapsed in seconds
% <stime> is [start end]

if ~exist('X','var')

  % internal constants
  noiselevel = 1;

  % generate predictors
  X = randn(n,p);

  % generate true weights
  htrue = randn(p,b);

  % generate data
  y = X*htrue;  % n x b
  y = y + bsxfun(@times,noiselevel*std(y,[],1),randn(size(y)));

end

% do it
stime = [];
stime(1) = str2double(unix_wrapper('date +%s.%N',0));
tic;
switch flag
case 1

  % standard approach to ridge regression
  cache1 = X'*y;  % p x b
  cache2 = X'*X;  % p x p
  h = zeros(p,b,length(hyper));
  for ii=1:length(hyper)
    h(:,:,ii) = inv(cache2 + hyper(ii)*eye(p))*cache1; 
  end

case 2

  % new approach
  [h,alphas] = fracridge(X,hyper,y);

case 3

  % new ALT approach
  [h,alphas] = fracridge(X,hyper,y,[],1);

end
etime = toc;
stime(2) = str2double(unix_wrapper('date +%s.%N',0));
