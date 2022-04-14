% dataname = 'semioccluded_l[0.00,-0.50,0.00]_r[1.57,0.00,3.14]_v[0.43,0.41,0.81]_s[256]_l[256]_gs[1.00]_conf.hdf5';
dataname = 'bunny_l[0.00,-0.50,0.00]_r[1.57,0.00,3.14]_v[0.80,0.53,0.81]_s[256]_l[256]_gs[1.00]_conf.hdf5';

%%
datainfo = h5info(dataname);

cameraGridNormals = h5read(dataname,'/cameraGridNormals');
cameraGridPoints = h5read(dataname,'/cameraGridPoints');
cameraGridPositions = double(h5read(dataname,'/cameraGridPositions'));
cameraGridSize = h5read(dataname,'/cameraGridSize');
cameraPosition = h5read(dataname,'/cameraPosition');
data = h5read(dataname,'/data');
deltaT =  h5read(dataname,'/deltaT');
hiddenVolumePosition = h5read(dataname,'/hiddenVolumePosition');
hiddenVolumeRotation = h5read(dataname,'/hiddenVolumeRotation');
hiddenVolumeSize = h5read(dataname,'/hiddenVolumeSize');
isConfocal = h5read(dataname,'/isConfocal');
laserGridNormals = h5read(dataname,'/laserGridNormals');
laserGridPoints = h5read(dataname,'/laserGridPoints');
laserGridPositions = h5read(dataname,'/laserGridPositions');
laserGridSize = h5read(dataname,'/laserGridSize');
laserPosition = h5read(dataname,'/laserPosition');
t = h5read(dataname,'/t');
t0 = h5read(dataname,'/t0');
%%
data = squeeze(data);
M = size(data,1); N = size(data,2);
data = double(squeeze(data(:,:,2,:)));

cdist = sqrt(sum((cameraGridPositions - reshape(cameraPosition,[1,1,3])).^2,3));
ldist = sqrt(sum((laserGridPositions - reshape(laserPosition,[1,1,3])).^2,3));
tdist = squeeze(ceil((cdist + ldist) * (1/deltaT)));
first_zero = zeros(M,N,4);
for j = 1:N
    for i = 1:M
        data(i,j,:) = circshift(data(i,j,:), - tdist(i,j));
    end
end

%% Downsample to fit the setting
data = (data(:,:,1:2:end) + data(:,:,2:2:end)) / 2;

% note that the source code of confocal NeTF only use the one-way
% transients of the NLOS setting, i.e. photons only travel from the object
% and hit the wall.
% While in most traditional methods like LCT, DLCT, Phasor Field, f-k, the
% transients are two-way, i.e. photons start from the wall, hit the object
% then reflected to wall again.

%%
data = data(:,:,1:1024); % only use part of the transients to save memory
data = permute(data,[3,1,2]);
cameraGridPositions_re = zeros(3, M * N);
for i = 1:1:M
    for j = 1:1:N
        cameraGridPositions_re(:, (i - 1) * N + j) = reshape(cameraGridPositions(i,j,:),3,1);
    end
end
cameraGridPositions = cameraGridPositions_re;
save zaragoza_semioccluded_256_preprocessed.mat cameraGridPoints cameraGridPositions cameraGridSize cameraPosition data deltaT hiddenVolumePosition hiddenVolumeSize
