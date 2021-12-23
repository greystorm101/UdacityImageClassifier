import argparse
import json
import utils
import torch


def predict(image_path, model, topk=5, gpu=True):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    
    Inputs:
        image_path (str): Path to image to classify
        model: model to be used for classification
        topk (int): How many results of top k predictions will be returned
        gpu (bool): If true, will try to run prediction on gpu.
    Outputs:
        top_p: percentages of top predictions
        top_class: classes of top predictions
    '''
    if (gpu):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device("cpu")
        
    print("Running with device: "+ str(device))

    model.to(device)
    image = utils.process_image(image_path)
    image = image.to(device)
    
    model.eval()
    
    logps = model(image)
    ps = torch.exp(logps)
    top_p, top_class = ps.topk(topk)
    
    with torch.no_grad():
        output = model.forward(image)
        ps = torch.exp(output)
        top_ps, top_classes = ps.topk(5, dim=1)
        
    return top_p, top_class

    
def main():
    """
    Predicts flower name from image as well as probability of that prediction
    """
    parser = argparse.ArgumentParser(description='Predict parser')
    
    #Add args
    #TODO remove flag requirement
    parser.add_argument('image_path', type = str, 
                        help = "Path to image to be predicted.")
    parser.add_argument('checkpoint', type = str,
                        help = "Path to checkpoint file to be predicted with")
    parser.add_argument('--top_k', action="store", type = int, required = False, default = 3,
                        help = "Number of predictions to be shown. Default: 3")
    parser.add_argument('--category_name', type = str,  required = False, default = "cat_to_name.json",
                        help = "json file with mapping of numbers to categories. Default: cat_to_name.json")
    parser.add_argument('--gpu', action = 'store_true', default = False,
                        help = "Flag to enable training on GPU if available. Default: False")

    args = parser.parse_args()
    
    #Load    
    model = utils.load_checkpoint(args.checkpoint)
    
    #Predict
    top_p, top_class = predict(args.image_path, model, topk = args.top_k, gpu = args.gpu)
    
    cat_to_name = utils.name_mapping(args.category_name)
    
    #Print results
    print("\nTop "+ str(args.top_k) + " Class(es):\n------")
    perc = top_p[0]
    perc = perc.to("cpu")
    perc = perc.detach().numpy()
    
    index = 0
    for elm in top_class[0]:
        elm = elm.item(); #Tensor magic
        dic_friendly = str(int(elm+1)) #Theres an off by 1 error
        name = cat_to_name[dic_friendly]
        print(name +": "+ str(perc[index]) + "%")
        index += 1
    
if __name__ == "__main__":
    main()