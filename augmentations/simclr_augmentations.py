import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.v2 as transforms

from torchvision.transforms.v2.functional import InterpolationMode

def get_meerkat_transforms(selected_transforms):
    s = 1 / 8.0  # Strength parameter for ColorJitter. Based on SimCLR strength parameter for supervised learning

    color_jitter = transforms.ColorJitter(0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s)
    gaussian = transforms.GaussianBlur(kernel_size=23) #Based on SimCLR, 10% of the image size, sigma = 0.1, 2.0


    # Define all the possible transformations
    transform_dict = {
        'random_resized_crop': transforms.RandomResizedCrop(224, scale=(0.8, 1.0), antialias=True, interpolation=InterpolationMode.BICUBIC),
        'horizontal_flip': transforms.RandomHorizontalFlip(),
        'gaussian_blur': transforms.RandomApply([gaussian], p=0.5),
        'color_jitter': transforms.RandomApply([color_jitter], p=0.8)
    }
    
    # Select the transforms based on input
    selected_transform_list = [transform_dict[transform] for transform in selected_transforms if transform in transform_dict]
    
    # Compose the selected transforms
    meerkat_transforms = transforms.Compose(selected_transform_list)
    
    return meerkat_transforms, transform_dict

if __name__ == "__main__":
    selected_transforms = ['random_resized_crop', 'gaussian_blur']
    meerkat_transforms = get_meerkat_transforms(selected_transforms)
    print(meerkat_transforms)