image_width = 128
image_height = 128
dataset_path = './aptos2019-blindness-detection/'
split_ratio = 0.2
seed = 42

alexnet_weights_path = 'bvlc_alexnet.npy'
num_classes = 5

imgnet_rgb_mean = [123.68, 116.779, 103.939]
imgnet_rgb_mean_resnet = [0.485, 0.456, 0.406]
imgnet_rgb_stddev_resnet = [0.229, 0.224, 0.225]


batch_size = 32
num_epochs = 20
lr = 1e-4