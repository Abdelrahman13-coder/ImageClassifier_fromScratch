import torch
import torchvision
import argparse
import json
import numpy as np
from PIL import Image
import torch.nn.functional as F

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('image_path', metavar='image_path', type=str, default='flowers/test/10/image_07090.jpg')
    parser.add_argument('--class_names', action='store', dest='class_names', type=str, default='cat_to_name.json')
    parser.add_argument('checkpoint', metavar='checkpoint', type=str, default='train_checkpoint.pth')
    parser.add_argument('--top_k', action='store', dest="top_k", type=int, default=5)
    parser.add_argument('--gpu', action='store_true', default=False)
    return parser.parse_args()

def load_checkpoint(filepath):
    checkpoint = torch.load(filepath, map_location=lambda storage, loc: storage)
    model = getattr(torchvision.models, checkpoint['pretrained_model'])(pretrained=True)
    model.input_size= checkpoint['input_size']
    model.output_size = checkpoint['output_size']
    model.learning_rate = checkpoint['learning_rate']
    model.hidden = checkpoint['hidden_units']
    model.learning_rate = checkpoint['learning_rate']
    model.classifier= checkpoint['classifier']
    model.epochs = checkpoint['epochs']
    model.class_to_idx = checkpoint['class_to_idx']
    model.optimizer= checkpoint['optimizer']
    model.load_state_dict(checkpoint['state_dict'])
    return model


def process_image(image):
    resize = 256
    crop_size = 224
    (w,h) = image.size

    if h > w:
        h = int(max(h * resize / w, 1))
        w = int(resize)
    else:
        w = int(max(w * resize / h, 1))
        h = int(resize)

  
    img = image.resize((w, h))

    left = (w - crop_size) / 2
    top = (h - crop_size) / 2
    right = left + crop_size
    bottom = top + crop_size
    img = img.crop((left, top, right, bottom))

    img = np.array(img)
    img = img / 255
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img = (img - mean) / std
    img = np.transpose(img, (2, 0, 1))
    return img


def predict(image_path, model, top_k, gpu):
    if gpu:
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    model.to(device)
    
    image = Image.open(image_path)
    image = process_image(image)
    image = torch.from_numpy(image).float()
    image = image.unsqueeze_(0)

    with torch.no_grad():
        output = model.forward(image.cuda())
    prob = F.softmax(output.data, dim=1)
    probs = np.array(prob.topk(top_k)[0][0])
    idx_to_class = {val: key for key, val in model.class_to_idx.items()}
    flowers = [np.int(idx_to_class[each]) for each in np.array(prob.topk(top_k)[1][0])]

    return probs, flowers, device


def load_names(class_names_file):
    with open(class_names_file) as file:
        class_names = json.load(file)
    return class_names


def main():
    args = parse_args()
    image_path = args.image_path
    checkpoint = args.checkpoint
    class_names = args.class_names
    top_k = args.top_k
    gpu = args.gpu

    model = load_checkpoint(checkpoint)
    probs, flowers, device = predict(image_path, model, top_k, gpu)
    class_names = load_names(class_names)
    labels = [class_names[str(index)] for index in flowers]

    print("prediction for image {}\n{}\n{}".format(image_path,labels,probs))
    for i in range(len(labels)):
        print("{} - {} with a probability of {}".format((i+1), labels[i], probs[i]))


if __name__ == "__main__":
    main()