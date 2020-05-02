% define
ns = [100];                           % number of data points (n)
ps = [2 5 10 50 100 500];             % number of predictors (p)
noiselevel = 1;                       % noise level in the data
ncorr = 2;                            % ncorr*p pairs of predictors will be combined
standardalphas = [0 10.^(-4:.5:5.5)]; % alpha grid for standard approach
fracs = 0:.05:1;                      % frac grid
numreps = 5;                          % number of simulations (at each combination)

% run
etime = [];  % elapsed time
for ii=1:length(ns)
  n = ns(ii);
  for jj=1:length(ps)
    for rr=1:numreps

      % generate predictors
      X = randn(n,ps(jj));
      for rep=1:round(ncorr*ps(jj))
        ix0 = ceil(rand(1,2)*ps(jj));
        X(:,ix0(1)) = sum(X(:,ix0),2) + randn(n,1);
      end
      X = calczscore(X,1);

      % generate true weights
      htrue = randn(size(X,2),1);

      % generate data
      y = X*htrue;
      y = y + noiselevel*std(y)*randn(size(y));
    
      % split into train and test
      Xtrain = X(1:round(n/2),:);
      Xtest = X(round(n/2)+1:end,:);
      ytrain = y(1:round(n/2));
      ytest = y(round(n/2)+1:end);

      % do it
      figureprep([100 100 700 550]);
      ax1 = []; touserng = [];
      for qq=3:-1:1  % 1 is standard, 2 is DOF, 3 is frac

        switch qq
        case {1 2}
      
          % standard approach to ridge regression
  if 0
          tic;
          cache1 = Xtrain'*ytrain;
          cache2 = Xtrain'*Xtrain;
          h = zeros(ps(jj),length(standardalphas));  % p x alphas
          for p=1:length(standardalphas)
            h(:,p) = inv(cache2 + standardalphas(p)*eye(size(Xtrain,2)))*cache1; 
          end
          etime(ii,jj,qq) = toc;
  else
          tic;
          h = fracridge(Xtrain,standardalphas,ytrain,[],1);
          etime(ii,jj,qq) = toc;
  end

          switch qq
          case 1
            touse = -log10(standardalphas);  % the x-axis values are -log10(alpha)
          case 2
            [u,s,v] = svd(Xtrain,0);
            ds = diag(s).^2;
            dof1 = [];
            for zz=1:length(standardalphas)
              dof1(zz) = sum(ds./(ds+standardalphas(zz)));
            end
            touse = dof1;                   % the x-axis values are dof
          end

        case 3
      
          % new approach
          tic;
          [h,alphas] = fracridge(Xtrain,fracs,ytrain);
          etime(ii,jj,qq) = toc;
          touse = fracs;

        end
        touserng(qq,:) = [min(touse) max(touse)];
        if qq==2
          touserng(qq,1) = 0;  % make DOF start at 0
        end

        % calculate training and testing error
        trainerr = calccod(Xtrain*h,repmat(ytrain,[1 length(touse)]),1,[],0);
        testerr =  calccod(Xtest*h, repmat(ytest,[1 length(touse)]),1,[],0);

        % plot
        subplot(3,3,qq); hold on;
        yyaxis left;
        plot(touse,trainerr,'ro-');
        ylabel('Training {\itR}^2 (%)');
        set(gca,'YColor','r')
        ax1(qq,1) = gca;
        yyaxis right;
        plot(touse,testerr,'ro-','Color',[.3 .3 1]);
        ylabel('Testing {\itR}^2 (%)');
        set(gca,'YColor',[.3 .3 1]);
        [mmx,iix] = max(testerr);
        scatter(touse(iix),mmx,'bo','filled');
        ax1(qq,2) = gca;
        yyaxis left;

        % plot more
        subplot(3,3,qq+3); hold on;
        plot(touse,vectorlength(h,1),'ro-');
        ylabel('Vector length of solution');
        ax1(qq,3) = gca;

        % plot more
        subplot(3,3,qq+6); hold on;
        plot(touse,h');
        ylabel('Beta weight');
        ax1(qq,4) = gca;
      
      end
    
      % enforce same y-axis for rows 2 and 3
      linkaxes(ax1(:,3),'y');
      linkaxes(ax1(:,4),'y');

      % do some last-minute plotting
      for qq=1:3  % 3 different columns

        % finish up
        for ss=[1 4 7]  % R2, vector length, beta weight
          subplot(3,3,(qq-1)+ss); hold on;
          if qq==3
            xlim([0 1]);
          else
            xlim(touserng(qq,:));
          end
          switch qq
          case 1  % standard approach
            if ss==1
              yyaxis left;
            end
            h0 = straightline(-log10(alphas),'v','k-');  % show where frac is in the -log10(alpha) space
            set(gca,'XTickLabel',-get(gca,'XTick'));
            xlabel('Log_{10}(alpha)');
          case 2  % DOF
            dof2 = [];
            for zz=1:length(alphas)
              dof2(zz) = sum(ds./(ds+alphas(zz)));
            end
            h0 = straightline(dof2,'v','k-');           % show where frac's DOFs would be
            xlabel('Degrees of freedom');
          case 3  % frac approach
            h0 = straightline(fracs,'v','k-');
            set(gca,'XTick',0:0.2:1);
            xlabel('Fraction');
          end
          set(h0,'Color',[.7 .7 .7]);
        end

      end

      figurewrite(sprintf('figure_n%02d_p%02d_rep%d',ii,jj,rr),[],-1,'simfigures');
    
    end

  end
  
end
