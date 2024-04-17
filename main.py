import os
import torch
import pandas as pd

from torchvision.io import read_image
from torchvision.ops.boxes import masks_to_boxes
from torchvision import tv_tensors
from torchvision.transforms.v2 import functional as F

import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torchvision.transforms import v2 as T
import matplotlib.pyplot as plt
from torchvision.utils import draw_bounding_boxes, draw_segmentation_masks
from engine import train_one_epoch, evaluate
from torchvision.io import read_image
import utils


# DATA_DIR = "/home/anicet/projects/starter-problem/DATASET/BH-POOLS-SPLIT"
DATA_DIR = "/home/ahounkanrin/swimming-pool-detection/DATASET/BH-POOLS"


class SwimmingPoolDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, csv_file, transforms):
        self.root_dir = root_dir
        self.data_frame = pd.read_csv(csv_file)
        self.transforms = transforms
        
        # load all image files, sorting them to
        # ensure that they are aligned
        # self.imgs = list(sorted(os.listdir(os.path.join(root, "PNGImages"))))
        # self.masks = list(sorted(os.listdir(os.path.join(root, "PedMasks"))))

    def __getitem__(self, idx):
        # load images and masks

        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_path = os.path.join(self.root_dir, self.data_frame.iloc[idx, 1])
        mask_path = os.path.join(self.root_dir, self.data_frame.iloc[idx, 2])

        # img_path = os.path.join(self.root, "PNGImages", self.imgs[idx])
        # mask_path = os.path.join(self.root, "PedMasks", self.masks[idx])

        img = read_image(img_path)
        mask = read_image(mask_path)
        # instances are encoded as different colors
        obj_ids = torch.unique(mask)
        # first id is the background, so remove it
        obj_ids = obj_ids[1:]
        num_objs = len(obj_ids)

        # split the color-encoded mask into a set
        # of binary masks
        masks = (mask == obj_ids[:, None, None]).to(dtype=torch.uint8)

        # get bounding box coordinates for each mask
        boxes = masks_to_boxes(masks)

        # there is only one class
        labels = torch.ones((num_objs,), dtype=torch.int64)

        image_id = idx
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        # suppose all instances are not crowd
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)

        # Wrap sample and targets into torchvision tv_tensors:
        img = tv_tensors.Image(img)

        target = {}
        target["boxes"] = tv_tensors.BoundingBoxes(boxes, format="XYXY", canvas_size=F.get_size(img))
        target["masks"] = tv_tensors.Mask(masks)
        target["labels"] = labels
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def __len__(self):
        return len(self.data_frame)
    

# Plot an example of image-mask pair
def plot_img_mask_pair(idx=10):

    df = pd.read_csv("train.csv")
    image = read_image(os.path.join(DATA_DIR, df.iloc[idx, 1])) # load image number idx in train dataset
    mask = read_image(os.path.join(DATA_DIR, df.iloc[idx, 2]))  # load mask number idx in train dataset

    plt.figure(figsize=(16, 8))
    plt.subplot(121)
    plt.title("Image")
    plt.imshow(image.permute(1, 2, 0))
    plt.subplot(122)
    plt.title("Mask")
    plt.imshow(mask.permute(1, 2, 0))
    plt.savefig(f"img_mask_pair_{idx}.png")
    plt.show()

# plot_img_mask_pair(10)

def get_model_instance_segmentation(num_classes):
    # load an instance segmentation model pre-trained on COCO
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(weights="DEFAULT")

    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    # now get the number of input features for the mask classifier
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    # and replace the mask predictor with a new one
    model.roi_heads.mask_predictor = MaskRCNNPredictor(
        in_features_mask,
        hidden_layer,
        num_classes
    )

    return model


def get_transform(train):
    transforms = []
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))
    transforms.append(T.ToDtype(torch.float, scale=True))
    transforms.append(T.ToPureTensor())
    return T.Compose(transforms)


def main_train():
    # train on the GPU or on the CPU, if a GPU is not available
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # our dataset has two classes only - background and person
    num_classes = 2
    # use our dataset and defined transformations

    dataset_train = SwimmingPoolDataset(DATA_DIR, "train.csv", get_transform(train=True))
    dataset_val = SwimmingPoolDataset(DATA_DIR, "train.csv", get_transform(train=False))

    # split the dataset in train and test set
    indices = torch.randperm(len(dataset_train)).tolist()
    val_set_size = int(0.2*len(dataset_train))
    dataset = torch.utils.data.Subset(dataset_train, indices[:-val_set_size])
    dataset_val = torch.utils.data.Subset(dataset_val, indices[-val_set_size:])

    # define training and validation data loaders
    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=2,
        shuffle=True,
        num_workers=4,
        collate_fn=utils.collate_fn
    )

    data_loader_val = torch.utils.data.DataLoader(
        dataset_val,
        batch_size=1,
        shuffle=False,
        num_workers=4,
        collate_fn=utils.collate_fn
    )

    # get the model using our helper function
    model = get_model_instance_segmentation(num_classes)

    # move model to the right device
    model.to(device)

    # construct an optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(
        params,
        lr=0.005,
        momentum=0.9,
        weight_decay=0.0005
    )

    # and a learning rate scheduler
    lr_scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=3,
        gamma=0.1
    )

    # let's train it just for 2 epochs
    num_epochs = 50
    for epoch in range(num_epochs):
        # train for one epoch, printing every 10 iterations
        train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=10)
        # update the learning rate
        lr_scheduler.step()
        # evaluate on the test dataset
        evaluate(model, data_loader_val, device=device)



    df = pd.read_csv("test.csv")
    image = read_image(os.path.join(DATA_DIR, df.iloc[10, 1])) # load image number idx in train dataset
    # mask = read_image(os.path.join(DATA_DIR, df.iloc[10, 2]))  # load mask number idx in train dataset

    # image = read_image("data/PennFudanPed/PNGImages/FudanPed00046.png")
    eval_transform = get_transform(train=False)

    model.eval()
    with torch.no_grad():
        x = eval_transform(image)
        # convert RGBA -> RGB and move to device
        x = x[:3, ...].to(device)
        predictions = model([x, ])
        pred = predictions[0]


    image = (255.0 * (image - image.min()) / (image.max() - image.min())).to(torch.uint8)
    image = image[:3, ...]
    pred_labels = [f"pedestrian: {score:.3f}" for label, score in zip(pred["labels"], pred["scores"])]
    pred_boxes = pred["boxes"].long()
    output_image = draw_bounding_boxes(image, pred_boxes, pred_labels, colors="red")

    # masks = (pred["masks"] > 0.7).squeeze(1)
    # output_image = draw_segmentation_masks(output_image, masks, alpha=0.5, colors="blue")


    plt.figure(figsize=(12, 12))
    plt.imshow(output_image.permute(1, 2, 0))
    plt.savefig("predictions.png")


if __name__ == "__main__":
    main_train()