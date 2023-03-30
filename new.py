import os
import torch
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import ToTensor
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
import torch.utils.model_zoo as model_zoo


class CustomDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        self.annotations = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        img_path = os.path.join(self.root_dir, self.annotations.iloc[index, 0])
        # Load image and convert to grayscale
        image = Image.open(img_path).convert("L")

        classes = ['node']
        class_dict = {c: i for i, c in enumerate(classes)}

        # Convert class label to one-hot vector
        class_label = self.annotations.iloc[index, 1]
        one_hot_label = [0] * len(classes)
        one_hot_label[class_dict[class_label]] = 1
        y_label = torch.tensor(one_hot_label)

        # Extract bounding box coordinates and convert to tensor
        xmin = self.annotations.iloc[index, 2]
        ymin = self.annotations.iloc[index, 3]
        xmax = self.annotations.iloc[index, 4]
        ymax = self.annotations.iloc[index, 5]
        boxes = torch.tensor([[xmin, ymin, xmax, ymax]], dtype=torch.float32)

        # Create target dictionary
        labels = torch.tensor([y_label], dtype=torch.int64)
        target = {"boxes": boxes, "labels": labels}

        # Apply transform to image if specified
        if self.transform:
            image = self.transform(image)

        return image, target

def get_faster_rcnn(num_classes):
    ## load an instance segmentation model pre-trained on COCO
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=False)

    # get the number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features

    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    backbone = torchvision.models.mobilenet_v2(weights="DEFAULT").features
    backbone.out_channels = 2048
    anchor_generator = AnchorGenerator(sizes=((32, 64, 128, 256, 512),),
                                   aspect_ratios=((0.5, 1.0, 2.0),))
    roi_pooler = torchvision.ops.MultiScaleRoIAlign(featmap_names=['0'],
                                                    output_size=7,
                                                    sampling_ratio=2)
    model = FasterRCNN(backbone,
                   num_classes=num_classes,
                   rpn_anchor_generator=anchor_generator,
                   box_roi_pool=roi_pooler)
    
    # get the number of output features from the backbone
    return model

def main():
    device = torch.device('cuda') if torch.cuda.is_available else torch.device('cpu')
    num_classes = 2
    csv_file = r'C:\Users\ASUS\Desktop\deeplearning-project\output-og-anno.csv'
    root_dir = 
    dataset = CustomDataset()


