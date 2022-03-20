function show_maps(D,cc)
% SHOW_MAPS(D,CC)
%   Show cross-sectional views around maximum-valued voxel for given CC-th
%   volume in D.

switch length(size(D))
    case 4
        % Permute 1st and 2nd dimensions
        D = permute(D,[2 1 3 4]);
        % Pick only selected component
        t = squeeze(D(:,:,:,cc));
    case 3
        % Permute 1st and 2nd dimensions
        t = permute(D,[2 1 3]);
end
% Find voxel with maximum value
[val, idx] = max(abs(t(:)));
[y x z] = ind2sub(size(t),idx);

% Create new figure with cross-sectional views
% centered on maximum-valued voxel
figure
subplot(1,3,1)  % Sagital view
I = transpose(squeeze(t(:,x,:)));
imagesc(I,max(abs(I(:)))*[-1 1]), axis xy equal tight
subplot(1,3,2)  % Coronal view
I = fliplr(transpose(squeeze(t(y,:,:))));
imagesc(I,max(abs(I(:)))*[-1 1]), axis xy equal tight
subplot(1,3,3)  % Axial view
I = fliplr(squeeze(t(:,:,z)));
imagesc(I,max(abs(I(:)))*[-1 1]), axis xy equal tight