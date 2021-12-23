
import argparse
from torchvision import transforms,datasets,models
import torch
import model
import utils

def main():
    parser = argparse.ArgumentParser(description='Training parser')
    
    #Add args
    parser.add_argument('data_directory', type = str,
                        help = "Path to directory with data. Default = 'flowers'")
    parser.add_argument('--save_dir', action="store", required = False,
                        help = "If provided, saves network to given filename")
    parser.add_argument('--arch', action="store", required = False, default = "vgg",
                        help = "Architecture used for pre-trained network. Options: vgg, alex. Default: vgg")
    parser.add_argument('--learning_rate', type = float, default = 0.003,
                        help = "Learning rate for training the model. Default: 0.003")
    parser.add_argument('--hidden_units', type = int, default = 816,
                        help = "Hidden units in the first layer of the network. Default: 816")
    parser.add_argument('--epochs', type=int, default = 20,
                       help = "Number of epochs to train with. Default:20")
    parser.add_argument('--gpu', action = 'store_true', default = False,
                        help = "Flag to enable training on GPU if available. Default: False")

    args = parser.parse_args()
    print(args)
    
    #Make sure arch is valid
    m_types = ["vgg", "alex"]
    if args.arch not in m_types:
        print("Error: "+args.arch+" not a valid option, defaulting to vgg")
        args.arch = "vgg"
    
    #Prep data
    train_dataloader, test_dataloader, valid_dataloader, train_set = utils.data_prep(args.data_directory)
  
    #Train model
    my_model = model.define_model(hidden_units = args.hidden_units, arch = args.arch)
    my_model = model.train_model(my_model, train_dataloader, test_dataloader,
                              gpu = args.gpu, epochs = args.epochs, learning_rate = args.learning_rate)
    #Test model
    model.test_model(my_model, valid_dataloader, test_dataloader, gpu = args.gpu)

    #save checkpoint
    #TODO check if we should save
    utils.save_checkpoint(my_model, args.save_dir, args.arch, train_set, args.hidden_units)
    
if __name__ == "__main__":
    main()