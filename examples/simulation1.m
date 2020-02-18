% define
ns = [100 1000 10000];
ps = [2 5 10 50 100 500 1000 5000];
ss = [1 100];
noiselevel = 1;
standardlambdas = 10.^(-4:.5:6);
gammas = 0:.05:1;

% run
etime = [];
for ii=1:length(ns)
  for jj=1:length(ps)
    for kk=1:length(ss)

      % generate regressors
      X = repmat(randn(ns(ii),1),[1 ps(jj)]) + ss(kk)*randn(ns(ii),ps(jj));
      X = calczscore(X,1);

      % generate true weights
      htrue = randn(size(X,2),1);

      % generate data
      y = X*htrue;
      y = y + noiselevel*std(y)*randn(size(y));

      % split into train and test
      Xtrain = X(1:round(end/2),:);
      Xtest = X(round(end/2)+1:end,:);
      ytrain = y(1:round(end/2));
      ytest = y(round(end/2)+1:end);

      % do it
      figureprep([100 100 600 600]);
      ax1 = [];
      for qq=1:2

        switch qq
        case 1
          tic;
          cache1 = Xtrain'*y;
          cache2 = Xtrain'*Xtrain;
          h = [];
          for p=1:length(standardlambdas)
            h(:,1,p) = inv(cache2 + standardlambdas(p)*eye(size(Xtrain,2)))*cache1;
          end
          etime(ii,jj,kk,qq) = toc;
          touse = -log(standardlambdas);
        case 2
          tic;
          [h,lambdas] = ridgeregressiongamma(Xtrain,gammas,ytrain);
          etime(ii,jj,kk,qq) = toc;
          touse = gammas;
        end

        % calculate training and testing error
        trainerr = calccod(Xtrain*squish(h,2),repmat(ytrain,[1 length(touse)]),1,[],0);
        testerr =  calccod(Xtest*squish(h,2),repmat(ytest,[1 length(touse)]),1,[],0);

        % plot
        subplot(3,2,(qq-1)+1); hold on;
        yyaxis left;
        plot(touse,trainerr,'ro-');
        ylabel('Training R^2 (%)');
        set(gca,'YColor','r')
        yyaxis right;
        plot(touse,testerr,'ro-','Color',[.3 .3 1]);
        ylabel('Testing R^2 (%)');
        set(gca,'YColor',[.3 .3 1]);
        ax1(qq) = gca;

        % plot more
        subplot(3,2,(qq-1)+3); hold on;
        plot(touse,flatten(vectorlength(h,1)),'ro-');
        if qq==2
          ax = axis;
          straightline(-log(lambdas1),'v','k-');
          axis(ax);
        end
        ylabel('Vector length of solution');

        % plot more
        subplot(3,2,(qq-1)+5); hold on;
        plot(touse,squish(h,2)');
        ylabel('Beta weight');

      end
      linkaxes(ax1,'y');
      figurewrite(sprintf('figure%02d',nn),[],[],'~/Dropbox/KKTEMP/sim');

    end

  end

end



elapsed time:
figureprep;
plot(etime);
figurewrite('time',[],[],'~/Dropbox/KKTEMP/sim');




%% %%%%%%%%%%%%%%%%%%%%%%%%%% JUNK


%     for p=1:50
%       i1 = ceil(rand*size(X,2));
%       i2 = ceil(rand*size(X,2));
%       X(:,i1) = X(:,i1) + sign(randn)*X(:,i2) + randn(size(X,1),1);
%     end
%    X = calczscore(X,1);
    %     trainerr = sum((Xtrain*squish(h,2) - repmat(ytrain,[1 length(touse)])).^2,1);
    %     testerr =  sum((Xtest*squish(h,2) - repmat(ytest,[1 length(touse)])).^2,1);
