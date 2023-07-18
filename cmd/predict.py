import argparse
import numpy as np
import json

from PIL import Image

import torch
from torchvision import models

parser = argparse.ArgumentParser()
parser.add_argument('image_path', type=str)
parser.add_argument('checkpoint', type=str)
parser.add_argument('--top_k', default=1, type=int)
parser.add_argument('--category_names', default='cat_to_name.json', type=str)
parser.add_argument('--gpu', action='store_true')

args = parser.parse_args()
image_path = args.image_path
checkpoint = args.checkpoint
top_k = args.top_k
category_names = args.category_names
gpu = args.gpu


def load_checkpoint(checkpoint):
    model_checkpoint = torch.load(checkpoint)

    model_arch = model_checkpoint['arch']
    if model_arch == 'alexnet':
        model = models.alexnet(weights=models.AlexNet_Weights.DEFAULT)
    elif model_arch == 'densenet121':
        model = models.densenet121(weights=models.DenseNet121_Weights.DEFAULT)
    else:
        print('The model architecture is not supported, terminating the program..')
        exit(1)

    model.classifier = model_checkpoint['classifier']
    model.load_state_dict(model_checkpoint['state_dict'])
    model.class_to_idx = model_checkpoint['mapping']
    epochs = model_checkpoint['epochs']
    lr = model_checkpoint['lr']
    return model, epochs, lr


def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    width, height = image.size

    if width < height:
        image = image.resize((256, 256*height//width))
    else:
        image = image.resize((256*width//height, 256))

    width, height = image.size
    reduce = 224
    left = (width - reduce)/2
    top = (height - reduce)/2
    right = left + 224
    bottom = top + 224
    im = image.crop((left, top, right, bottom))

    np_image = np.array(im)/255
    np_image -= np.array([0.485, 0.456, 0.406])
    np_image /= np.array([0.229, 0.224, 0.225])

    np_image = np_image.transpose((2, 0, 1))
    return np_image


def predict(image_path, model, gpu, topk=5):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    image = Image.open(image_path)
    image = torch.from_numpy(process_image(image)).type(
        torch.FloatTensor)  # type: ignore

    with torch.no_grad():
        if gpu:
            model.to('cuda')
            image = image.to('cuda')
        else:
            model.to('cpu')

        # we're using unsqueeze to add a batch dimension so that it wpuld be compatible with the model
        model.eval()
        log_ps = model.forward(image.unsqueeze(0))

    ps = torch.exp(log_ps)
    values, indices = ps.topk(topk)

    idx_to_class = {val: key for key, val in model.class_to_idx.items()}

    indices = indices.cpu()
    values = values.cpu()

    classes = [idx_to_class[idx] for idx in indices.numpy()[0].tolist()]

    return values, classes


def main():
    with open(category_names, 'r') as f:
        cat_to_name = json.load(f)

    model, epochs, lr = load_checkpoint(checkpoint)
    values, classes = predict(image_path, model, gpu, top_k)
    classes = [cat_to_name[cls] for cls in classes]

    for img_pr, img_class in zip(values.numpy()[0], classes):
        print(f'{img_class}: {img_pr:.2f}')


if __name__ == '__main__':
    main()
