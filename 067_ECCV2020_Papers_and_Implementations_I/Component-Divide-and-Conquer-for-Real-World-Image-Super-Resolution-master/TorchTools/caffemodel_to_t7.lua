require 'loadcaffe'
require 'xlua'
require 'optim'

—- modify the path 

prototxt = 'MTCNNv1/model/'
binary = '/home/fanq15/pconvert_caffe_to_pytorch/vgg16.caffemodel'

net = loadcaffe.load(prototxt, binary, 'cudnn')
net = net:float() —- essential reference https://github.com/clcarwin/convert_torch_to_pytorch/issues/8
print(net)

torch.save('/home/fanq15/convert_caffe_to_pytorch/vgg16_torch.t7', net)




