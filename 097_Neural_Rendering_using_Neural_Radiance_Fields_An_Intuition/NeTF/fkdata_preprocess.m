load('tof.mat');
load('meas_180min.mat');
wall_size = 2; 
width = wall_size / 2;
M = 256; % we only use 256 x 256 in our experiments
meas = meas(1:(512/M):end, 1:(512/M):end, :);
tofgrid = tofgrid(1:(512/M):end, 1:(512/M):end);
deltaT = 32e-12;
c = 3e8;
for ii = 1:size(meas, 1)
    for jj = 1:size(meas,2)
        meas(ii, jj, :) = circshift(meas(ii, jj, :), [0, 0, -floor(tofgrid(ii, jj) / (deltaT*1e12))]);
    end
end  
meas = (meas(:,:,1:2:end) + meas(:,:,2:2:end)) / 2; 
meas = permute(meas, [3,1,2]);
meas = meas(1:1:512,:,:);

x = linspace(1,-1,M);
z = linspace(1,-1,M);
[X,Z] = meshgrid(x,z);
grid = cat(3, X,Z);
grid = reshape(grid, [M * M, 2]);
grid = [grid(:,1), zeros(M*M,1), grid(:,2)]';
cameraGridPositions = grid;

cameraGridSize = [wall_size;wall_size];
cameraGridPoints = [M;M];
cameraPosition = [0,0,0];
data = double(meas);
hiddenVolumePosition = [0,-1.4,0];
hiddenVolumeSize = 2;

save fk_bike_meas_180_min_256_preprocessed cameraGridPoints cameraGridPositions cameraGridSize cameraPosition data deltaT hiddenVolumePosition hiddenVolumeSize

