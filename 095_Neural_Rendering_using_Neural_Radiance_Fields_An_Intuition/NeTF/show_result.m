addpath('./model')
load predicted_volume4_256.mat

%% show 2D images from different views
albedo = volume_rho .* volume;
albedo = flip(albedo,1);
albedo = flip(albedo,3);
img_XOY = squeeze(max(albedo,[],3));
img_XOZ = squeeze(max(albedo,[],2));
img_YOZ = squeeze(max(albedo,[],1));
figure
subplot(1,3,1)
imshow(img_XOY,[])
title('side view - XOY')
subplot(1,3,2)
imshow(img_XOZ,[])
title('front view - XOZ')
subplot(1,3,3)
imshow(img_YOZ,[])
title('side view - YOZ')
%% show 3D density
threshold = graythresh(volume);
threshold = threshold * 1;

figure()
fv = isosurface(volume,threshold);
p = patch(fv);
isonormals(volume,p) 
p.FaceColor = 'red';
p.EdgeColor = 'none';
daspect([1 1 1])
view(3); 
view(60,40)
axis tight 
light
xlabel('Y')
ylabel('X')
zlabel('Z')