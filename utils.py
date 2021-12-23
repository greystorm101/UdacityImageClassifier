import torch
from torchvision import transforms,datasets,models
import numpy as np
from PIL import Image
import json
import model

"""
File for various utility functions
"""
def save_checkpoint(model, filepath, arch, train_set, hidden_units):
    """
    Saves a model checkpoint to a file
    
    Inputs:
        model: the model to be saved
        filepath (str): Filename to save checkpoint to
        arch (str): architecture used for base of model
        train_set: train set from data_prep
        hidden_units(int): number of hidden units in the checkpoint
    """
    #Save the checkpoint 
    model.class_to_idx = train_set.class_to_idx
    
    
    checkpoint = {'arch':arch,
              'classifier': model.classifier,
              'state_dict': model.state_dict(),
              'class_mapping': model.class_to_idx,
               'hidden_units':hidden_units}

    torch.save(checkpoint, filepath)
    
    print("Saved checkpoint to "+filepath)


def load_checkpoint(filepath):
    
    checkpoint = torch.load(filepath)
    arch = checkpoint['arch']
    
    myModel = model.define_model(hidden_units = checkpoint['hidden_units'], arch=arch)
    
    #Freeze the params
    for param in myModel.parameters():
        param.requires_grad = False
        
    myModel.classifier = checkpoint['classifier']
    #model.load_state_dict(checkpoint['state_dict'])
    myModel.class_to_idx = checkpoint['class_mapping']
    
    return myModel

def name_mapping(filename = 'cat_to_name.json'):
    """
    Generates name mapping dictionary
    Input:
        filename (str) : Name of json file containing mappping
    Output:
        cat_to_name (dict): Dictionary mapping number to a name
    """
    with open(filename, 'r') as f:
        cat_to_name = json.load(f)
    return cat_to_name


def data_prep(data_dir = 'flowrs'):
    """
    Preps data for model from data directory
    
    Inputs:
        data_dir (string): directory with 3 subdirectories (train, valid, and test)
    Outputs:
        train_dataloader, test_dataloader, valid_dataloader: Dataloaders for 3 image subsets
        train_set: used for saving checkpoint
        
    """
    #Define subdirs
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'

    #Define your transforms for the training, validation, and testing sets
    normalize_array_1 = [0.485, 0.456, 0.406]
    normalize_array_2 = [0.229, 0.224, 0.225]

    #Training Set
    train_transforms = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(30),
        transforms.RandomResizedCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(normalize_array_1, 
                            normalize_array_2)
    ])

    #Testing Set
    test_transforms = transforms.Compose([
        transforms.CenterCrop(225),
        transforms.Resize([224,224]),
        transforms.ToTensor(),
        transforms.Normalize(normalize_array_1, 
                            normalize_array_2)
    ])

    #Validation Set
    valid_transforms = transforms.Compose([
        transforms.CenterCrop(225),
        transforms.Resize([224,224]),
        transforms.ToTensor(),
        transforms.Normalize(normalize_array_1, 
                            normalize_array_2)
    ])

    #Load the datasets with ImageFolder
    train_set = datasets.ImageFolder(train_dir, transform = train_transforms)
    test_set = datasets.ImageFolder(test_dir, transform = test_transforms)
    valid_set = datasets.ImageFolder(valid_dir, transform = valid_transforms)

    # Using the image datasets and the trainforms, define the dataloaders
    train_dataloader = torch.utils.data.DataLoader(train_set, batch_size=64, shuffle=True)
    test_dataloader = torch.utils.data.DataLoader(test_set, batch_size=64, shuffle=True)
    valid_dataloader = torch.utils.data.DataLoader(valid_set, batch_size=64, shuffle=True)

    return train_dataloader, test_dataloader, valid_dataloader, train_set


def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns not an Numpy array
    '''
    
    means = np.array([0.485, 0.456, 0.406])
    std_dev = np.array([0.229, 0.224, 0.225])
    
    l_size = 500, 256
    nl_size = 256, 500
    
    with Image.open(image) as im:
        
        longer = im.size[0] > im.size[1]
        if longer:
            im.thumbnail(l_size)
        else:
            im.thumbnail(nl_size)
        
        #Crop
        width = im.width; height = im.height
        upper = (width - 224) /2
        left = (width - 224) /2
        lower = upper + 224
        right = left + 224
        
        im_crop = im.crop((left,upper,right,lower))
        #Convert to 0-1 scale
        np_image = np.array(im_crop) / 256
        
        #normalize
        np_image = (np_image - means) / std_dev
        
        np_image = np_image.transpose((2, 0, 1))
    
        return torch.FloatTensor([np_image])

