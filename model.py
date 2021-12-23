#For model and model training/predicting functions
from torchvision import transforms,datasets,models
import torch
from torch import nn, optim


def define_model(hidden_units = 816, arch = "vgg"):
    """
    Defines model
    
    Input:
        hidden_units (int): Number of nodes in first hidden layer
        arch (str): Type of pretrained model to use. Supports vgg or alex.
    Output:
        model
    """

    #Make sure you're running at least python 3 or this will break
    '''
    if (m_type == "vgg"):
        model = models.vgg16(pretrained=True)
        input_num = 25088
    elif (m_type == "alex"):
        model = models.alexnet(pretrained=True)
        input_num = 9216
    else:
        print(m_type + " is not a supporrted model (vgg, alex, or resnet). Defaulting to vgg")
        model = models.vgg16(pretrained=True)
        input_num = 25088
    '''
    if (arch == "vgg"):
        model = models.vgg16(pretrained=True)
        input_num = 25088
    elif (arch == "alex"):
        model = models.alexnet(pretrained=True)
        input_num = 9216
    else:
        print(arch + " is not a supporrted model (vgg, or alex). Defaulting to vgg")
        model = models.vgg16(pretrained=True)
        input_num = 25088
    
    #Freeze the params
    for param in model.parameters():
        param.requires_grad = False

    default_hl1 = hidden_units; default_hl2 = 408; default_hl3 = 204
    output_num = 102
    dropout = 0.05

    classifier = nn.Sequential(nn.Linear(input_num, default_hl1),
                          nn.ReLU(),
                          nn.Dropout(dropout),

                          nn.Linear(default_hl1, default_hl2),
                          nn.ReLU(),
                          nn.Dropout(dropout),

                          #nn.Linear(default_hl2, default_hl3),
                          #nn.ReLU(),
                          #nn.Dropout(dropout),     

                          nn.Linear(default_hl2, output_num),
                          nn.LogSoftmax(dim=1))

    model.classifier = classifier
    
    return model


def train_model(model, train_dataloader, test_dataloader, gpu = True, epochs = 10, learning_rate = 0.001):
    """
    Trains a model
    
    inputs:
        model: defined model to train
        train_dataloader: dataloader with training daya
        test_dataloader: dataloader with testing data
        gpu (boolean): If true, trains on gpu. Default = True
        epochs (int): Number of epochs to train with. Default = 10
        learning_rate (float): Learning rate for model. Default = 0.001
    
    outputs:
        model: trained model
    """
    #Put on correct device to run
    if (gpu):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device("cpu")
        
    print("Beginning training with device: "+ str(device))

    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate)
    model.to(device)

    train_losses, test_losses = [], []
    for e in range(epochs):
        running_loss = 0
        for train_images, train_labels in train_dataloader:
            train_images, train_labels = train_images.to(device), train_labels.to(device)

            log_ps = model.forward(train_images)
            loss = criterion(log_ps, train_labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
        else:
            test_loss = 0
            accuracy = 0

            with torch.no_grad():
                for test_images, test_labels in test_dataloader:
                    #IDK if i need this too but probably
                    test_images, test_labels = test_images.to(device), test_labels.to(device)

                    log_ps = model.forward(test_images)
                    test_loss += criterion(log_ps, test_labels)

                    ps = torch.exp(log_ps)
                    top_p, top_class = ps.topk(1, dim=1)
                    equals = top_class == test_labels.view(*top_class.shape)
                    accuracy += torch.mean(equals.type(torch.FloatTensor))

            train_losses.append(running_loss/len(train_dataloader))
            test_losses.append(test_loss/len(test_dataloader))

            print("Epoch: {}/{}.. ".format(e+1, epochs),
                  "Training Loss: {:.3f}.. ".format(running_loss/len(train_dataloader)),
                  "Test Loss: {:.3f}.. ".format(test_loss/len(test_dataloader)),
                  "Test Accuracy: {:.3f}".format(accuracy/len(test_dataloader)))
    return model


def test_model(model, valid_dataloader, test_dataloader, gpu = True):
    """
    Prints test loss and accuracy
    Inputs:
        model: trained model
        valid_dataloader: dataloader for validation data
        test_dataloader: dataloader for validation data
        gpu (boolean): If true, uses gpu. Defaults to True.
    """
    criterion = nn.NLLLoss()

    test_loss = 0
    accuracy = 0

    #Put on correct device to run
    if (gpu):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device("cpu")
        
    print("Begining testing with device: "+ str(device))
    
    model.to(device)

    with torch.no_grad():
        for test_images, test_labels in valid_dataloader:
            test_images, test_labels = test_images.to(device), test_labels.to(device)
            log_ps = model.forward(test_images)

            test_loss += criterion(log_ps, test_labels)

            ps = torch.exp(log_ps)
            top_p, top_class = ps.topk(1, dim=1)
            equals = top_class == test_labels.view(*top_class.shape)
            accuracy += torch.mean(equals.type(torch.FloatTensor))

    print("Test Loss: {:.3f}.. ".format(test_loss/len(test_dataloader)),
          "Test Accuracy: {:.3f}".format(accuracy/len(test_dataloader)))
            