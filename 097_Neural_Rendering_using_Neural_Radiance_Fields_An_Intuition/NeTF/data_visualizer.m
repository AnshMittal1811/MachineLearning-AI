%% parameters

db_anme = 'fk_bike_meas_180_min_256_preprocessed.mat';
output_name = 'fk_dragon_meas_180_min_256_preprocessed_rep.avi';

%% run visualizer
addpath('./data')
load(db_anme)

if ~isempty(output_name)
    v = VideoWriter(output_name);
    v.FrameRate = 15;
    open(v)
end

for i=1 : size(data,1)

    I = reshape(data(i,:,:),256,256);
    I = uint8(10*I);
    imshow(I,[])

    if ~isempty(output_name)
        writeVideo(v,I)
    end
end 

if ~isempty(output_name)
    close(v)
end