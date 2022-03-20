% Load the AAL atlas map: one 3D volume in a 3D matrix
% The values in the AAL atlas are numerical labels that identify different
% regions of the brain.
% Assumes 'aal_labels.nii' file is in the current folder
AAL_map = spm_read_vols(spm_vol(fullfile(pwd,'aal_labels.nii')));

% show_maps() is a support function provided along with this script for the
% 2014 MLSP Competition.
show_maps(AAL_map,1)
set(gca,'clim',[0 max(AAL_map(:))])
colormap colorcube