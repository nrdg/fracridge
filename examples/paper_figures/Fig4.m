%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% Simple setup

% define
stimfile = 'nsd_stimuli.hdf5';
expfile = 'nsd_expdesign.mat';
nsess = 37;                % how many of the first N NSD sessions to consider
subjix = 1;                % which subject [81 104 83]
ng  = 25;                  % number of elements in contrast grid (ng x ng)
npx = 17;                  % number of pixels that make up one grid element (npx x npx)
stimdeg = 8.4;             % size of stimulus in degrees of visual angle
slicestart = 40;           % which slice to start from
slicenum = 10;             % how many

% load
exp1 = load(expfile);

%% Do some stimulus pre-processing

% determine vector of all of the 73k IDs that are involved in the data we will load
allimix = unique(exp1.subjectim(subjix,exp1.masterordering(1:750*nsess)));

% load and prepare images
ims = zeros(425,425,length(allimix));
for p=1:length(allimix)
  statusdots(p,length(allimix));
  im0 = h5read(stimfile,'/imgBrick',[1 1 1 allimix(p)],[3 425 425 1]);
  im0 = permute(im0,[3 2 1]);
  if p==1
    imwrite(im0,'sampleimage.png');
    imwrite(rgb2gray(im0),'sampleimagegray.png');
  end
  im = single(rgb2gray(im0));  % convert to grayscale and to single format
  im = (im/255).^2;            % convert to [0,1] and square to match display gamma
  ims(:,:,p) = im;
end

% compute a "contrast grid" from the images
imagecon = zeros(ng,ng,length(allimix));  % 25 x 25 x images
for rowix=1:ng
  statusdots(rowix,ng);
  for colix=1:ng
    rowii = (rowix-1)*npx + (1:npx);
    colii = (colix-1)*npx + (1:npx);
    imagecon(rowix,colix,:) = std(squish(ims(rowii,colii,:),2),[],1);  % standard deviation of pixels
  end
end

% clean
clear ims;

% save a cached version of <imagecon> for convenience
save('stimuluscache.mat','imagecon');

%% Do some experimental-design preparation

% when do the trials for each image occur?
trialgroup = cell(1,length(allimix));  % 1 x images with trial indices
temp = exp1.subjectim(subjix,exp1.masterordering(1:750*nsess));
for p=1:length(allimix)
  trialgroup{p} = find(temp==allimix(p));
end

%% Load NSD betas

% load data
data = zeros(81,104,10,750,nsess);  % X x Y x Z x 750 trials x sessions
for p=1:nsess
  fprintf('sess %d...',p);
  file0 = sprintf('betas_session%02d.hdf5',p);
  temp = h5read(file0,'/betas',[1 1 slicestart 1],[Inf Inf slicenum 750]);
  temp = double(temp)/300;                  % convert to double and then convert to percent signal change
  data(:,:,:,:,p) = temp;                   % record
end

% z-score the data
data = calczscore(data,4);
data(isnan(data)) = 0;  % some voxels in some sessions have invalid...

% average across trials and re-z-score
data2 = zeros([sizefull(data,3) length(trialgroup)]);  % X x Y x Z x images
for p=1:length(trialgroup)
  data2(:,:,:,p) = mean(data(:,:,:,trialgroup{p}),4);
end
data = data2; clear data2;
data = calczscore(data,4);
data(isnan(data)) = 0;  % some voxels in some sessions have invalid...

%% Do some training/testing preparation

% calculate the training/testing split
[testix,~,trainix] = picksubset(1:size(data,4),[5 1]);

% deal with data
ytrain = squish(data(:,:,:,trainix),3)';  % trials x voxels
ytest  = squish(data(:,:,:,testix),3)';   % trials x voxels

% prep the stim
stim = calczscore(squish(imagecon,2),2);  % 25*25 x imagees
imwrite(uint8(255*cmaplookup(reshapesquare(stim(:,1)),min(stim(:,1)),max(stim(:,1)),0,gray(256))),'samplecon.png');

% deal with stimulus
Xtrain = stim(:,trainix)';    % trials x 625 regressors  
Xtest  = stim(:,testix)';     % trials x 625 regressors

% clean
clear data;

%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% PREP

% load
ytest = ytest(:,1:81*104);     % just get first slice!
ytrain = ytrain(:,1:81*104);   % just get first slice!

%% FIT

% define
standardalphas = [0 10.^(-4:.5:5.5)];    % alpha grid for standard approach
fracs = 0:.05:1;                         % frac grid

% calc
p = size(Xtrain,2);
n = size(Xtrain,1);
b = size(ytrain,2);
f = length(standardalphas);

% do it
etime = []; stime = [];
for flag=1:3
  stime(flag,1) = str2double(unix_wrapper('date +%s.%N',0));
  tic;
  switch flag
  case 1

    % standard approach to ridge regression
    cache1 = Xtrain'*ytrain;  % p x b
    cache2 = Xtrain'*Xtrain;  % p x p
    h1 = zeros(p,b,length(standardalphas));
    for ii=1:length(standardalphas)
      h1(:,:,ii) = inv(cache2 + standardalphas(ii)*eye(p))*cache1;
    end

  case 2

    % new approach
    [h2,alphas] = fracridge(Xtrain,fracs,ytrain);

  case 3

    % new ALT approach
    h3 = fracridge(Xtrain,standardalphas,ytrain,[],1);

  end
  etime(flag) = toc;
  stime(flag,2) = str2double(unix_wrapper('date +%s.%N',0));
end

% massage
allh = cat(4,single(permute(h1,[1 3 2])),single(h2),single(h3));  % 625 x 21hyper x voxels x 3approaches

% clean
clear h1 h2 h3;

%% CALC R2

trainR2 = zeros(b,length(standardalphas),3);  % voxels x 21hyper x 3approaches
testR2  = zeros(b,length(standardalphas),3);
for typ=1:3
  for ii=1:length(standardalphas), ii
    trainR2(:,ii,typ) = calccod(Xtrain*squish(allh(:,ii,:,typ),2),ytrain,1,[],0,0);
    testR2(:,ii,typ) = calccod(Xtest*squish(allh(:,ii,:,typ),2),ytest,1,[],0,0);
  end
end

%% VISUALIZE

mkdirquiet('figures');

% map of R2
whsl = 1;
[mm,ii] = max(testR2,[],2);
for typ=1:3
  vol = reshape(mm(:,:,typ),[81 104 1]);
  imwrite(uint8(255*makeimagestack(vol(:,:,whsl),[0 20],0)),hot(256),sprintf('figures/R2_typ%d.png',typ));
  imwrite(uint8(255*cmaplookup(makeimagestack(vol(:,:,whsl),[],0),0,20,[],hot(256))),sprintf('R2_typ%d_thresh.png',typ), ...
    'Alpha',uint8(255*(makeimagestack(vol(:,:,whsl),[],0)>20/4)))
  vol = reshape(ii(:,:,typ),[81 104 1]);
  if ismember(typ,[1 3])
    vol = 22-vol;  % flip
  end
  imwrite(uint8(255*makeimagestack(vol(:,:,whsl),[1 21],0)),copper(256),sprintf('figures/maxindex_typ%d.png',typ));
end

% T1
a1 = load_untouch_nii('T1_to_func1pt8mm.nii.gz');
imwrite(uint8(255*makeimagestack(double(a1.img(:,:,40)),1,0)),gray(256),sprintf('figures/T1.png'));

% pick a voxel
if 0
  vol = reshape(mm(:,:,1),[81 104 1]);
  figure; imagesc(vol(:,:,1));
end

% for many voxels, visualize the progression of rf weights
rowiis = 1:81;
colii = 10;
voxelix = [];
mkdirquiet('figures/rf/');
mxs = [];
for zz=1:length(rowiis)
  voxelix(zz) = sub2ind([81 104 1],rowiis(zz),colii,1);
  for typ=1:3
    ok = reshape(allh(:,:,voxelix(zz),typ),[25 25 21]);
    if ismember(typ,[1 3])
      ok = flipdim(ok,3);
    end
    mx = max(abs(ok(:)));
    mxs(zz,typ) = mx;
    if mx==0
      mx = 1;
    end
  end
  for typ=1:3
    ok = reshape(allh(:,:,voxelix(zz),typ),[25 25 21]);
    if ismember(typ,[1 3])
      ok = flipdim(ok,3);
    end
    finalmx = max(mxs(zz,:),[],2);
    if finalmx==0
      finalmx = 1;
    end
    imwrite(uint8(255*makeimagestack(ok,[-finalmx finalmx+(2*finalmx)*1/255],[],[1 21])),[cmapsign4(255); 1 1 1], ...
      sprintf('figures/rf/rf_typ%d_num%02d.png',typ,zz));
  end
end

% record some numbers
mxs([41 52],:)
% 
% ans =
% 
%         0.0817369669675827        0.0817369669675827        0.0817369669675827
%          0.156348064541817         0.156348064541817         0.156348064541817

figureprep([100 100 300 300]); hold on;
imagesc((0:20)/20,[0 1]); colormap(copper(256)); colorbar;
figurewrite('colorbarcopper',[],-1,pwd);

figureprep([100 100 300 300]); hold on;
imagesc(1); colormap(cmapsign4(255)); colorbar;
figurewrite('colorbarcmapsign4',[],-1,pwd);

% compute DOF for standard alphas
[u,s,v] = svd(Xtrain,0);
ds = diag(s).^2;
dof1 = [];
for zz=1:length(standardalphas)
  dof1(zz) = sum(ds./(ds+standardalphas(zz)));
end

%% Big figure

% do it
for zzz=1:length(voxelix)
  figureprep([100 100 700*.75 550*.75]);
  ax1 = []; touserng = []; firstyrng = []; secondyrng = [];
  for qq=3:-1:1  % 1 is standard, 2 is DOF, 3 is frac

    switch qq
    case {1 2}
  
      % standard approach to ridge regression

      switch qq
      case 1
        touse = -log10(standardalphas);  % the x-axis values are -log10(alpha)
      case 2
        touse = dof1;                   % the x-axis values are dof
      end

    case 3
  
      % new approach
      touse = fracs;

    end
    touserng(qq,:) = [min(touse) max(touse)];
    if qq==2
      touserng(qq,1) = 0;  % make DOF start at 0
    end

    % extract quantities
    switch qq
    case 1
      wh0 = 1;
    case 2
      wh0 = 1;
    case 3
      wh0 = 2;
    end
    trainerr = trainR2(voxelix(zzz),:,wh0)';
    testerr =  testR2(voxelix(zzz),:,wh0)';
    h = allh(:,:,voxelix(zzz),wh0);
    alphas0 = alphas(:,voxelix(zzz));

    % plot
    subplot(3,3,qq); hold on;
    yyaxis left;
    plot(touse,trainerr,'ro-');
    ylabel('Training {\itR}^2 (%)');
    set(gca,'YColor','r');
    firstyrng(qq,:) = ylim;
    ax1(qq,1) = gca;
    yyaxis right;
    plot(touse,testerr,'ro-','Color',[.3 .3 1]);
    ylabel('Testing {\itR}^2 (%)');
    set(gca,'YColor',[.3 .3 1]);
    [mmx,iix] = max(testerr);
    scatter(touse(iix),mmx,'bo','filled');
    secondyrng(qq,:) = ylim;
    ax1(qq,2) = gca;
    yyaxis left;

    % plot more
    subplot(3,3,qq+3); hold on;
    plot(touse,vectorlength(h,1),'ro-');
    ylabel('Vector length');
    ax1(qq,3) = gca;

    % plot more
    subplot(3,3,qq+6); hold on;
    plot(touse,h');
    ylabel('Beta weight');
    ax1(qq,4) = gca;
  
  end

  % enforce same y-axis
  for qq=1:3
    subplot(3,3,qq);
    yyaxis left;
    ylim([min(firstyrng(:,1)) max(firstyrng(:,2))]);
    yyaxis right;
    ylim([min(secondyrng(:,1)) max(secondyrng(:,2))]);
  end
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
        h0 = straightline(-log10(alphas0),'v','k-');  % show where frac is in the -log10(alpha) space
        set(gca,'XTick',-4:2:4);
        set(gca,'XTickLabel',-get(gca,'XTick'));
        xlabel('Log_{10}alpha');
      case 2  % DOF
        dof2 = [];
        for zz=1:length(alphas0)
          dof2(zz) = sum(ds./(ds+alphas0(zz)));
        end
        h0 = straightline(dof2,'v','k-');           % show where frac's DOFs would be
        set(gca,'XTick',0:300:600);
        xlabel('Degrees of freedom');
      case 3  % frac approach
        h0 = straightline(fracs,'v','k-');
        set(gca,'XTick',0:.5:1);
        xlabel('Fraction');
      end
      set(h0,'Color',[.7 .7 .7]);
    end

  end

  figurewrite(sprintf('vx%03d',zzz),[],-1,'figures/inspections');
end
