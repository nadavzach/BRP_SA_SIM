import torch
import torchvision.models as models
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import numpy as np
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='save np arrays of mobilenet and alexnet weights disterbutions')
    parser.add_argument('--a_w', type=int,default=50, help='a width')
    parser.add_argument('--a_h', type=int,default=50, help='a height')
    parser.add_argument('--a_c', type=int,default=50, help='a channels')
    parser.add_argument('--b_h', type=int,default=50, help='b height')
    parser.add_argument('--path', type=str ,default='./a_b_mats', help='path')


    args = parser.parse_args()
    a_w = args.a_w
    a_h = args.a_h
    a_c = args.a_c
    b_w = a_c
    b_h = args.b_h
    path = args.path

    # Define the transform to be applied to the input images
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    print("loading CIFAR100 dataset... \n\n")
    # Load the CIFAR-100 dataset
    cifar100 = datasets.CIFAR100(root="./data", train=True, download=True, transform=transform)

    print("done! calculating mean variance and zero%..\n")

    mean = np.mean(cifar100.data / 255, axis=(0, 1, 2))
    mean = np.mean(mean)
    variance = np.var(cifar100.data / 255, axis=(0, 1, 2))
    variance = np.mean(variance)
    zero_percent = np.count_nonzero(cifar100.data == 0) / np.prod(cifar100.data.shape)
    print("done! saving b matrix in \n")

    values = np.random.normal(loc=mean,scale= np.sqrt(variance), size=b_w*b_h)
    values = np.reshape(values,(b_w,b_h))
    b_num_zeros = int(zero_percent * b_w * b_h )
    b_zero_indices = np.random.choice(b_w * b_h, b_num_zeros, replace=False)
    values.ravel()[b_zero_indices] = 0
    new_zero_per = np.count_nonzero(values == 0) / np.prod(values.shape)
    print("calc zero per = "+str(zero_percent)+", new zero per is "+str(new_zero_per)+" \n")

    np.savez_compressed("{}/{}.npz".format(path, 'CIFAR100_b_mat'), values)


    print("done! loading alexnet and mobilenet v3 small... \n")
    # Load the pre-trained AlexNet model
    models = [models.alexnet(pretrained=True),models.mobilenet_v3_small]


    print("done! claculating mean var and zero% for each... \n")
    # Put the model in evaluation mode
    for model in models:
        model.eval()
        # Loop through the layers of the model
        for name, module in model.named_modules():
            print("starting for "+name+"\n\n")
            if isinstance(module, torch.nn.Conv2d) or isinstance(module, torch.nn.Linear):
                # Calculate the mean and variance of the layer's weights
                mean = torch.mean(module.weight.data)
                variance = torch.var(module.weight.data)
                zero_per = 100*np.count_nonzero(module.weight.data == 0)/np.prod(module.weight.data.size())
                print("Layer {}: Mean={}, Variance={}".format(name, mean, variance))
                values = np.random.normal(loc=mean,scale= np.sqrt(variance),size= (a_w*a_h*a_c))
                values = np.reshape(values,(a_w,a_h,a_c))
                a_num_zeros = int(zero_percent * a_w * a_h*a_c )
                a_zero_indices = np.random.choice(a_w * a_h*a_c, a_num_zeros, replace=False)
                values.ravel()[a_zero_indices] = 0
                new_zero_per = np.count_nonzero(values == 0) / np.prod(values.shape)
                print("calc zero per = "+str(zero_percent)+", new zero per is "+str(new_zero_per)+" \n")
                np.savez_compressed("{}/{}.npz".format(path, 'a_mat_'+name), values)
    



