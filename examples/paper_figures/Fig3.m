% first, clear all and start the daemon:
%   perl cpumemorydaemon.pl 824 > bench.txt
% then, run the code.

%%

% define
baseb = 1000;  % number of targets
basep = 5000;  % number of predictors [this is critical]
basen = 5000;  % number of data points
baseh = 20;     % number of hyperparameters
standardalphas = 10.^(-4:.5:5.5);     % alpha grid for standard approach
fracs = .05:.05:1;                   % frac grid
numreps = 20;  % number of simulations

% define more
numb = [100 200 500 1000 2000 5000 10000 20000 30000 40000 50000];
nump = [50 100 200 500 1000 2000 5000 7500 10000 12500 15000];
numn = [100 200 500 1000 2000 5000 10000 15000 20000];
numh = [2 5 10 20 50 75 100];

% init
etime = cell(1,4);  % cell array of targets x (standard/frac/fracALT)
stime = cell(1,4);  % cell array of [start,finish] x targets x (standard/frac/fracALT)
mem   = cell(1,4);

% repeat simulations to average out variability
for rep=1:numreps

  % play with number of targets
  E = 1
  for ii=1:length(numb), ii
    [etime{E}(ii,1,rep),stime{E}(:,ii,1,rep),X,y] = Fig3_simhelper(basen,basep,numb(ii),1,standardalphas);
    [etime{E}(ii,2,rep),stime{E}(:,ii,2,rep)] = Fig3_simhelper(basen,basep,numb(ii),2,fracs,X,y);
    [etime{E}(ii,3,rep),stime{E}(:,ii,3,rep)] = Fig3_simhelper(basen,basep,numb(ii),3,standardalphas,X,y);
  end

  % play with number of predictors
  E = 2
  for ii=1:length(nump), ii
    [etime{E}(ii,1,rep),stime{E}(:,ii,1,rep),X,y] = Fig3_simhelper(basen,nump(ii),baseb,1,standardalphas);
    [etime{E}(ii,2,rep),stime{E}(:,ii,2,rep)] = Fig3_simhelper(basen,nump(ii),baseb,2,fracs,X,y);
    [etime{E}(ii,3,rep),stime{E}(:,ii,3,rep)] = Fig3_simhelper(basen,nump(ii),baseb,3,standardalphas,X,y);
  end

  % play with number of data points
  E = 3
  for ii=1:length(numn), ii
    [etime{E}(ii,1,rep),stime{E}(:,ii,1,rep),X,y] = Fig3_simhelper(numn(ii),basep,baseb,1,standardalphas);
    [etime{E}(ii,2,rep),stime{E}(:,ii,2,rep)] = Fig3_simhelper(numn(ii),basep,baseb,2,fracs,X,y);
    [etime{E}(ii,3,rep),stime{E}(:,ii,3,rep)] = Fig3_simhelper(numn(ii),basep,baseb,3,standardalphas,X,y);
  end

  % play with number of hyperparameters
  E = 4
  for ii=1:length(numh), ii
    standardalphas0 = 10.^linspace(-4,5.5,numh(ii));
    fracs0 = 1/numh(ii):1/numh(ii):1;
    [etime{E}(ii,1,rep),stime{E}(:,ii,1,rep),X,y] = Fig3_simhelper(basen,basep,baseb,1,standardalphas0);
    [etime{E}(ii,2,rep),stime{E}(:,ii,2,rep)] = Fig3_simhelper(basen,basep,baseb,2,fracs0,X,y);
    [etime{E}(ii,3,rep),stime{E}(:,ii,3,rep)] = Fig3_simhelper(basen,basep,baseb,3,standardalphas0,X,y);
  end

  % save
  saveexcept('simresults.mat',{'X' 'y'});

end

%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% load results
load('simresults.mat');

% read in the benchmark info
bfid = fopen('bench.txt');
b1 = textscan(bfid,'%f %f%s');
fclose(bfid);

% convert benchmark info
b1b = zeros(size(b1{1},1),2);  % entries x [time/mem (GiB)]
for p=1:size(b1{1},1)
  b1b(p,1) = b1{1}(p);
  switch b1{3}{p}
  case ''
    b1b(p,2) = b1{2}(p)/1024/1024;
  case 'm'
    b1b(p,2) = b1{2}(p)/1024;
  case 'g'
    b1b(p,2) = b1{2}(p);
  case 't'
    b1b(p,2) = b1{2}(p)*1024;
  otherwise
    die;
  end
end

% derive memory usage
for E=1:length(stime)
  for p=1:size(stime{E},2)
    for q=1:size(stime{E},3)
      for rr=1:size(stime{E},4)
        ix1 = b1b(:,1) > stime{E}(1,p,q,rr) & b1b(:,1) < stime{E}(2,p,q,rr);
        temp = b1b(ix1,2);
        mem{E}(p,q,1,rr) = max(temp-temp(1));  % max mem usage
        mem{E}(p,q,2,rr) = mean(temp-temp(1)); % mean
      end
    end
  end
end

% define
xlabels = {'Number of targets (\it{b})' 'Number of predictors (\it{p})' 'Number of data points (\it{n})' 'Number of hyperparameters (\it{f})'};
xquants = {numb nump numn numh};
xfix = [baseb basep basen baseh];

% make figure
cmap0 = {[1 0 0] [0 0 1] [1 .5 0]};     % naive RR is red circle, fracridge is blue diamond, rotated RR is orange square
markers = {'o' 'd' '*'};
figureprep([0 0 1000 500]); hold on;
neworder = [3 2 4 1];
for E=1:4
  ee = neworder(E);

  subplot(2,4,E); hold on;
  yy = mean(etime{ee},3);
  for rr=1:size(yy,3)
    for cc=[1 3 2]
      plot(xquants{ee},yy(:,cc,rr),[markers{cc} '-'],'LineWidth',1.5,'Color',cmap0{cc});
    end
  end
  xlabel(xlabels{ee});
  ylabel('Execution time (s)');
  ax = axis;
  axis([0 max(xquants{ee}) 0 max(yy(:))]);
  set(straightline(xfix(ee),'v','k-'),'Color',[.5 .5 .5]);
  if ee==4
    set(gca,'XTick',0:25:100);
  end

  subplot(2,4,4+E); hold on;
  yy = permute(mean(mem{ee}(:,:,1,:),4),[1 2 4 3]);
  yy2 = permute(mean(mem{ee}(:,:,2,:),4),[1 2 4 3]);
  for rr=1:size(yy,3)
    for cc=[1 3 2]
      plot(xquants{ee},yy(:,cc,rr),[markers{cc} '-'],'LineWidth',1.5,'Color',cmap0{cc});
      plot(xquants{ee},yy2(:,cc,rr),['--'],'LineWidth',1,'Color',cmap0{cc});
    end
  end
  xlabel(xlabels{ee});
  ylabel('Memory (GiB)');
  ax = axis;
  axis([0 max(xquants{ee}) 0 max(yy(:))]);
  set(straightline(xfix(ee),'v','k-'),'Color',[.5 .5 .5],'LineWidth',0.5);
  if ee==4
    set(gca,'XTick',0:25:100);
  end

end
figurewrite('benchmark',[],-1,'~/Dropbox/KKTEMP');
