% This is an example of how to load and display the functional (fMRI) and
% structural (sMRI) ICA maps, provided for the 2014 MLSP Competition, using
% MATLAB R2011a.
%
% It also includes an example of how to compute the correlation between the
% fMRI and sMRI maps, like was done in reference [2] below.
%
% This script requires the SPM toolbox to be installed. The code was
% tested with SPM 8. http://www.fil.ion.ucl.ac.uk/spm/software/spm8/
%
% The maps provided for the competition are a subset of the components
% discussed in the following publications:
% 
% 28 aggregate resting-state fMRI ICA maps:
% [1] E. Allen, et al, "A baseline for the multivariate comparison of
% resting state networks," Frontiers in Systems Neuroscience, vol. 5, p.
% 12, 2011.
% 
% 32 mean gray-matter concentration sMRI ICA maps:
% [2] Segall JM, et al (2012) Correspondence between structure and function
% in the human brain at rest. Front. Neuroinform. 6:10. doi: 10.3389/fninf.2012.00010 

%% Load fMRI maps
% Load 28 rs-fMRI ICA maps: one 3D volume per map, all in a single 4D matrix
% Assumes 'rs_fMRI_ica_maps.nii' file is in the current folder
Df = spm_read_vols(spm_vol(fullfile(pwd,'rs_fMRI_ica_maps.nii')));

% Load fMRI component numbers (as defined in reference [1] above)
fMRI_comp_ind = csvread('comp_ind_fMRI.csv',1,0);

% Select component to display:
%fMRI_ci = 6; % numbers 1..28
fMRI_ci = find(fMRI_comp_ind == 24); %numbers as defined in [1] above

% show_maps() is a simple support function provided along with this script
% for the 2014 MLSP Competition.
show_maps(Df,fMRI_ci)

%% Load sMRI maps
% Load 32 sMRI ICA maps: one 3D volume per map, all in a single 4D matrix
% Assumes 'gm_sMRI_ica_maps.nii' file is in the current folder
Ds = spm_read_vols(spm_vol(fullfile(pwd,'gm_sMRI_ica_maps.nii')));

% Load sMRI component numbers (as defined in reference [2] above)
sMRI_comp_ind = csvread('comp_ind_sMRI.csv',1,0);

% Select component to display:
%sMRI_ci = 9; % numbers 1..32
sMRI_ci = find(sMRI_comp_ind == 10); % numbers as defined in [2] above

% show_maps() is a support function provided along with this script for the
% 2014 MLSP Competition.
show_maps(Ds,sMRI_ci)

%% Correlation between fMRI and sMRI maps

% A simple way to find relationship between fMRI and sMRI features is by
% measuring linear correlation between the provided ICA maps from each
% modality.

% Reshape ICA maps to columns of a table:
sz = size(Df);
Dftab = reshape(Df,prod(sz(1:3)),sz(4));
sz = size(Ds);
Dstab = reshape(Ds,prod(sz(1:3)),sz(4));

% Create a mask to remove lines from the table which correspond to
% out-of-brain locations.
% This is useful for the correlation computation.

% Out-of-brain locations are those with zero-valued values:
msk = sum(abs(Dftab),2) ~= 0;

% Correlation between fMRI and sMRI maps:
cor_fs = corr(Dftab(msk,:),Dstab(msk,:)); % A 28x32 correlation matrix

% View correlations in an image:
figure
imagesc(cor_fs,[-.5 .5])
set(gca,'fontsize',5)
set(gca,'Xtick',1:length(sMRI_comp_ind),'Ytick',1:length(fMRI_comp_ind))
set(gca,'XtickLabel',sMRI_comp_ind,'YtickLabel',fMRI_comp_ind)
axis equal tight
colorbar