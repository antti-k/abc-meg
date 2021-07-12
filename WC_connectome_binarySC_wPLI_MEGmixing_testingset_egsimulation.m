clear;
% addpath(genpath('D:\MEGMOD\'));
% addpath(genpath('D:\DH7-H916-03\HIIT\'));
addpath(genpath('/Users/antti/matlab/bdtoolkit-2019a'));
addpath(genpath('/Users/antti/matlab/MEGMOD'));

%% Subject IDs

% subject_sets = {'S0003','S0006','S0011','S0034','S0039','S0116','S0121','S0124','S0125','S0128',...
%     'S0136','S0137','S0141','S0142','S0144','S0145','S0149','S0150','S0151','S0156','S0166','S0172','S0173'};

subject_sets = {'S0003'};

%% Initialise 

N=74;
pctval=90;
lefthem_idxs=[1:2:N*2];
righthem_idxs=[2:2:N*2];

% SC network

fname=strcat('/Users/antti/matlab/MEGMOD/data/SC_parc2k9_57HCP.mat');
load(fname,'avg_sc');
SC=avg_sc(lefthem_idxs,lefthem_idxs);
SC=SC+SC';

UTMAT=triu(ones(N),1);
ut_idxs=find(UTMAT(:));

pct_thresh=prctile(SC(ut_idxs),pctval);
SC_bin=double(SC>=pct_thresh);
Kij = SC_bin./(repmat(sum(SC_bin,2),[1,N]).*ones(N));

% MEG FC network

load('/Users/antti/matlab/MEGMOD/data/PY2LV_transvec','PY2LV_transvec');

fname=strcat('wPLImats_9.83Hz_46_PARC2k9_148parcs_wgroupmat.mat');
load(fname,'wPLImats_strength_groupmat1');
%fname = strcat('S0003_set01_wPLI_submat.mat');
%load(fname, 'wPLI_submat');

MEG_strength_wPLI_FC=wPLI_submat(lefthem_idxs,lefthem_idxs);

% Parameter values

m=5;
fc=9.83;
%fs=500;
fs=250;
ts=(1/fs)*10^3;

% mask_length=30;
mask_length=15;
mask_samples=mask_length*fs;
% data_length=630;
data_length=315;
data_samples=data_length*(fs*ts);

%% Original model - simulation
     
cycles=1;
D_all=cell(cycles,1);

MOD_strength_wPLI_FC=cell(cycles,1);
MODFIT_wPLI=zeros(cycles,2);

for cyc_idx=1:cycles

% Parameters from Wilson & Cowan (1972)    
        
Je = 2;
Ji = 0;
sys = WilsonCowanNet(Kij,Je,Ji);
sys.pardef=bdSetValue(sys.pardef,'wee',16);
sys.pardef=bdSetValue(sys.pardef,'wei',12);
sys.pardef=bdSetValue(sys.pardef,'wie',15);
sys.pardef=bdSetValue(sys.pardef,'wii',3);
sys.pardef=bdSetValue(sys.pardef,'k',2);
sys.pardef=bdSetValue(sys.pardef,'be',4);
sys.pardef=bdSetValue(sys.pardef,'bi',3.7);
sys.pardef=bdSetValue(sys.pardef,'taue',23.7);
sys.pardef=bdSetValue(sys.pardef,'taui',23.7);    
    
% Simulate model
 
tic;

tspan = [0 data_samples];               % Integration time domain
sol = bdSolve(sys,tspan,@ode45);        % Apply the ode45 solver
tplot = ts:ts:data_samples;             % Interpolation time points
Y = bdEval( sol,tplot);                  % Interpolate the solution

D=Y(1:size(Y,1)/2,:);
D_all{cyc_idx}=D;



wPLImags_mat=zeros(N,N,length(subject_sets));
    
for sub_idx=1:length(subject_sets)
                    
    % Source to parcels
    fname = char(strcat('/Users/antti/matlab/MEGMOD/data/MEG_subjectwise_data/sourceIDs/',subject_sets(sub_idx),'_set01__parc2009.csv'));
    source2parc=table2array(readtable(fname,'ReadVariableNames',0));           
                        
    % Forward operator
    fname = char(strcat('/Users/antti/matlab/MEGMOD/data/MEG_subjectwise_data/fixed_fwd_ops/',subject_sets(sub_idx),'_set01__py-fwd.csv'));
    fwd_mat=table2array(readtable(fname,'ReadVariableNames',0));
            
    % Inverse operator
    fname = char(strcat('/Users/antti/matlab/MEGMOD/data/MEG_subjectwise_data/fidweighted_inv_ops/',subject_sets(sub_idx),'_set01__parc2009.csv'));
    inv_mat=table2array(readtable(fname,'ReadVariableNames',0));
            
    % Linear mixing
    D_linmix = forwardinverse_modelling(D,source2parc,PY2LV_transvec,lefthem_idxs,fwd_mat,inv_mat);
             
    % Morlet filtering      
    D_linmix_filt = wavelet_filtering(D_linmix,m,fc,fs);           
            
    % Estimate Phase Locking Value (PLV)
    
    [~,wPLImags]=wPLI_calc(D_linmix_filt(:,mask_samples+1:end));           
            
    wPLImags(1:N+1:end)=0;
    wPLImags_mat(:,:,sub_idx)=wPLImags+wPLImags';         
            
    % disp(sub_idx);
            
end

MOD_strength_wPLI_FC{cyc_idx}=mean(wPLImags_mat,3);
        
% Performance measures
         
MODFIT_wPLI(cyc_idx,1)=sqrt(mean((MEG_strength_wPLI_FC(ut_idxs)-MOD_strength_wPLI_FC{cyc_idx}(ut_idxs)).^2));
MODFIT_wPLI(cyc_idx,2)=corr(MEG_strength_wPLI_FC(ut_idxs),MOD_strength_wPLI_FC{cyc_idx}(ut_idxs));

toc;
disp(cyc_idx);

end

%% Estimating mean FC matrices

MOD_strength_wPLI_FCmean=mean(reshape(cell2mat(MOD_strength_wPLI_FC'),[N,N,cycles]),3);
MODFIT_wPLI_mean(1)=sqrt(mean((MEG_strength_wPLI_FC(ut_idxs)-MOD_strength_wPLI_FCmean(ut_idxs)).^2));
MODFIT_wPLI_mean(2)=corr(MEG_strength_wPLI_FC(ut_idxs),MOD_strength_wPLI_FCmean(ut_idxs));

%% Determining statistical significance of correlation

perms=100;

% wPLI
CC_surr=zeros(perms,1);
RMSE_surr=zeros(perms,1);
for cc_idx=1:perms
    CC_surr(cc_idx)=corr(MEG_strength_wPLI_FC(ut_idxs),MOD_strength_wPLI_FCmean(ut_idxs(randperm(length(ut_idxs)))));
    RMSE_surr(cc_idx)=sqrt(mean((MEG_strength_wPLI_FC(ut_idxs)-MOD_strength_wPLI_FCmean(ut_idxs(randperm(length(ut_idxs))))).^2));
end
zCC_wPLI=abs((MODFIT_wPLI_mean(2)-mean(CC_surr))/std(CC_surr));
zRMSE_wPLI=abs((MODFIT_wPLI_mean(1)-mean(RMSE_surr))/std(RMSE_surr));

pval_CC_wPLI=2*(1-normcdf(zCC_wPLI));
pval_RMSE_wPLI=2*(1-normcdf(zRMSE_wPLI));

%% MEG and MOD matrices - comparison of MEG and MOD

% wPLI - Comparison of MEG and MOD

figure;
subplot(1,2,1)
imagesc(MEG_strength_wPLI_FC,[0,0.4]);
set(gca,'YDir','Normal','FontSize',18);
axis square;
xlabel('Region #');
ylabel('Region #');
colorbar('TickLabels',{'0','0.1','0.2','0.3','\geq 0.4'});
title('MEG');

subplot(1,2,2)
imagesc(MOD_strength_wPLI_FCmean,[0,0.4]);
set(gca,'YDir','Normal','FontSize',18);
axis square;
xlabel('Region #');
ylabel('Region #');
colorbar('TickLabels',{'0','0.1','0.2','0.3','\geq 0.4'});
title('MOD');

set(gcf,'Position',[100,100,1200,400]);

% saveas(gcf,'D:\MEGMOD\figs\MODFIT_WC_connectome_binarySC_wPLI_MEGmixing_k2_600s_500Hz_simresults_testingset_lefthem.png','png');

%% Scatter plots - comparison of MEG and MOD

% wPLI

figure;
plot(MEG_strength_wPLI_FC(ut_idxs),MOD_strength_wPLI_FCmean(ut_idxs),'o','MarkerSize',5);
set(gca,'FontSize',18);
h=lsline;
h.Color=[1,0,0];
h.LineWidth=2;
xlim([0,0.4]);
ylim([0,0.4]);
xlabel('MEG');
ylabel('MOD');
axis square;
set(gca,'XTick',[0:0.1:0.4],'YTick',[0:0.1:0.4]);

% saveas(gcf,'D:\MEGMOD\figs\scatterplot_WC_connectome_binarySC_wPLI_MEGmixing_k2_600s_500Hz_simresults_testingset_lefthem.png','png');

% '/Users/antti/matlab/
% saveas(gcf,'/Users/antti/matlab/scatterplot_WC_connectome_binarySC_wPLI_MEGmixing_k2_600s_500Hz_simresults_testingset_lefthem.png','png');

%% Histograms - comparison of MEG and MOD

% wPLI

figure;
subplot(1,2,1)
histogram(MEG_strength_wPLI_FC(ut_idxs),linspace(0,0.4,20));
set(gca,'YDir','Normal','FontSize',18);
axis square;
ylim([0,1000]);
xlabel('Strength');
ylabel('Count');
title('MEG');

subplot(1,2,2)
histogram(MOD_strength_wPLI_FCmean(ut_idxs),linspace(0,0.4,20));
set(gca,'YDir','Normal','FontSize',18);
axis square;
ylim([0,1000]);
xlabel('Strength');
ylabel('Count');
title('MOD');

set(gcf,'Position',[100,100,1200,400]);

% saveas(gcf,'D:\MEGMOD\figs\HIST_WC_connectome_binarySC_wPLI_MEGmixing_k2_600s_500Hz_simresults_testingset_lefthem.png','png');
