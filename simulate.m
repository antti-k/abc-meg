function out = simulate(new_wee)
addpath(genpath('/Users/antti/matlab/bdtoolkit-2019a'));
addpath(genpath('/Users/antti/matlab/MEGMOD'));

disp(class(new_wee));
disp(class(16));
wee = typecast(new_wee, 'double');
disp(class(wee));

N=74;
pctval=90;
lefthem_idxs=[1:2:N*2];
righthem_idxs=[2:2:N*2];
% subject_sets = {'S0003'};

%subject_sets = {'S0001','S0005','S0008','S0049','S0113','S0117','S0118','S0119','S0120','S0122','S0123','S0126','S0127','S0129','S0140','S0143','S0146','S0147','S0148','S0152','S0153','S0174','S0175'};
subject_sets = {'S0001','S0005','S0008','S0049','S0113','S0117','S0119','S0120','S0122','S0123','S0126','S0127','S0129','S0140','S0143','S0146','S0147','S0148','S0152','S0153','S0174','S0175'};


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

% fname=strcat('wPLImats_9.83Hz_46_PARC2k9_148parcs_wgroupmat.mat');
% load(fname,'wPLImats_strength_groupmat2');
fname = strcat('/Users/antti/matlab/MEGMOD/data/MEG_subjectwise_data/wPLI_submats/S0003_set01_wPLI_submat.mat');
load(fname, 'wPLI_submat');

MEG_strength_wPLI_FC=wPLI_submat(lefthem_idxs,lefthem_idxs);

% Parameter values

m=5;
fc=9.83;
%fs=500;
fs=250;
ts=(1/fs)*10^3;

% mask_length=30;
% mask_length=15;
% minimum
mask_length=5;
mask_samples=mask_length*fs;
% data_length=630;
% data_length=315;
% minimum
data_length=65;
data_samples=data_length*(fs*ts);

cycles=1;
D_all=cell(cycles,1);

MOD_strength_wPLI_FC=cell(cycles,1);
MODFIT_wPLI=zeros(cycles,2);

for cyc_idx=1:cycles

% Parameters from Wilson & Cowan (1972)    



Je = 2;
Ji = 0;
sys = WilsonCowanNet(Kij,Je,Ji);
%sys.pardef=bdSetValue(sys.pardef,'wee', 16);
sys.pardef=bdSetValue(sys.pardef,'wee', wee);
sys.pardef=bdSetValue(sys.pardef,'wei',12);
sys.pardef=bdSetValue(sys.pardef,'wie',15);
sys.pardef=bdSetValue(sys.pardef,'wii',3);
sys.pardef=bdSetValue(sys.pardef,'k',2);
sys.pardef=bdSetValue(sys.pardef,'be',4);
sys.pardef=bdSetValue(sys.pardef,'bi',3.7);
sys.pardef=bdSetValue(sys.pardef,'taue',23.7);
sys.pardef=bdSetValue(sys.pardef,'taui',23.7);    
    
% Simulate model
 

tspan = [0 data_samples];               % Integration time domain
sol = bdSolve(sys,tspan,@ode45);        % Apply the ode45 solver
tplot = ts:ts:data_samples;             % Interpolation time points
Y = bdEval( sol,tplot);                  % Interpolate the solution

D=Y(1:size(Y,1)/2,:);
D_all{cyc_idx}=D;



wPLImags_mat=zeros(N,N,length(subject_sets));
    
for sub_idx=1:length(subject_sets)
    disp(sub_idx);
                    
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
         
%MODFIT_wPLI(cyc_idx,1)=sqrt(mean((MEG_strength_wPLI_FC(ut_idxs)-MOD_strength_wPLI_FC{cyc_idx}(ut_idxs)).^2));
%MODFIT_wPLI(cyc_idx,2)=corr(MEG_strength_wPLI_FC(ut_idxs),MOD_strength_wPLI_FC{cyc_idx}(ut_idxs));


end

% out = wPLImags_mat;
out =   mean(wPLImags_mat,3);