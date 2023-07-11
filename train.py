import torch
from torch.utils.data import DataLoader
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from trail import train_ibll, test_ibll
from sklearn.metrics import precision_recall_curve, PrecisionRecallDisplay
from torchvision.ops import box_iou
import numpy as np
import matplotlib.pyplot as plt
plt.ion()  # Turn on interactive mode

def get_model(num_classes):
    model = fasterrcnn_resnet50_fpn(pretrained=True)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model

def collate_fn(batch):
    return tuple(zip(*batch))

def check_and_fix_bbox(bbox):
    """
    Ensures that the bbox has a positive width and height.
    If not, it adds a small epsilon value to make it positive.
    """
    eps = 1e-5
    if bbox[2] - bbox[0] <= 0:  # if width <= 0
        bbox[2] += eps
    if bbox[3] - bbox[1] <= 0:  # if height <= 0
        bbox[3] += eps
    return bbox

class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, images, targets):
        self.images = images
        self.targets = targets

    def __getitem__(self, idx):
        image = self.images[idx]['img']  # Assuming the 'img' item is a numpy array
        image = image.transpose((2, 0, 1))  # Change (H, W, C) to (C, H, W)
        image = image / 255.0  # Normalize to [0, 1]
        image = torch.from_numpy(image).float()  # Convert to tensor

        # Now, convert the target into the correct format
        boxes = self.targets[idx]
        boxes = torch.as_tensor(boxes, dtype=torch.float32)

        # Keep only the bounding boxes with positive area
        boxes = boxes[(boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1])]
        
        # Check if there are no valid bounding boxes
        if boxes.nelement() == 0:
            # Handle the case of having no valid bounding boxes
            boxes = torch.tensor([[0.0, 0.0, 1.0, 1.0]], dtype=torch.float32)  # A default box

        labels = torch.ones((len(boxes),), dtype=torch.int64)  # Assuming all objects belong to a single class
        target = {"boxes": boxes, "labels": labels}

        return image, target

    def __len__(self):
        return len(self.images)

def get_precision_recall(targets, predictions, threshold=0.5):
    gt_boxes = [t['boxes'].cpu().numpy() for t in targets]
    pred_boxes = [p['boxes'].cpu().numpy() for p in predictions]
    scores = [p['scores'].cpu().numpy() for p in predictions]

    # Flatten the lists
    gt_boxes = np.concatenate(gt_boxes, axis=0)
    pred_boxes = np.concatenate(pred_boxes, axis=0)
    scores = np.concatenate(scores, axis=0)

    # Calculate IoUs
    ious = box_iou(torch.tensor(gt_boxes), torch.tensor(pred_boxes))

    # Calculate true and false positives
    tp = (ious > threshold).sum(axis=0)
    fp = (ious <= threshold).sum(axis=0)

    # Calculate precision and recall
    precision = tp / (tp + fp)
    recall = tp / len(gt_boxes)

    # Convert precision and recall to averages
    precision = torch.mean(precision).item()
    recall = torch.mean(recall).item()

    return precision, recall, scores

# Prepare training and testing data
train_targets = [data['bboxes'] for data in train_ibll.data]
test_targets = [data['bboxes'] for data in test_ibll.data]

if __name__ == "__main__":
    dataset = CustomDataset(train_ibll.data, train_targets)
    data_loader = DataLoader(dataset, batch_size=2, shuffle=True, num_workers=4, collate_fn=collate_fn)

    test_dataset = CustomDataset(test_ibll.data, test_targets)
    test_data_loader = DataLoader(test_dataset, batch_size=2, shuffle=False, num_workers=4, collate_fn=collate_fn)

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print("CUDA available: ", torch.cuda.is_available())

    num_classes = 2
    model = get_model(num_classes)
    model.to(device)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)

    num_epochs = 25
    loss_values = []
    precision_values = []
    recall_values = []

    for epoch in range(num_epochs):
        model.train()
        i = 0    
        epoch_losses = []
        for images, targets in data_loader:
            images = list(image.to(device) for image in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            epoch_losses.append(losses.item())

            optimizer.zero_grad()
            losses.backward()
            optimizer.step()

            i += 1
            if i % 50 == 0:
                print(f"Iteration #{i} loss: {losses.item()}")

        # Calculate average loss for this epoch and save it
        avg_loss = sum(epoch_losses) / len(epoch_losses)
        loss_values.append(avg_loss)

        # Calculate precision and recall on the test set
        model.eval()
        all_targets = []
        all_predictions = []
        with torch.no_grad():
            for images, targets in test_data_loader:
                images = list(image.to(device) for image in images)
                predictions = model(images)
                all_targets.extend(targets)
                all_predictions.extend(predictions)

        try:
            precision, recall, _ = get_precision_recall(all_targets, all_predictions)
            precision_values.append(precision)
            recall_values.append(recall)
            print(f"Epoch {epoch}: Precision={precision}, Recall={recall}")
        except Exception as e:
            print(f"Exception during precision/recall calculation: {e}")

        # Save the model checkpoint
        if epoch % 2 == 0:
            torch.save(model.state_dict(), f"./models/model_aug_epoch_{epoch}.pth")

        print(f"Training...............................GPU engaged")
        print(f"[......................................................]") 
        print(f"[......................................................]") 
        print(f"[......................................................]")    
        print(f"Epoch #.............{epoch} loss: .....................{losses.item()}")

        # Save the loss values
        torch.save(loss_values, './models/loss_values.pth')

        # After training and validation for each epoch, plot the precision, recall and loss values
        plt.figure(figsize=(10, 8))
        plt.subplot(2, 1, 1)
        plt.plot(range(len(precision_values)), precision_values, label='Precision')
        plt.plot(range(len(recall_values)), recall_values, label='Recall')
        plt.xlabel('Epoch')
        plt.ylabel('Value')
        plt.title('Precision and Recall Curves')
        plt.legend()

        plt.subplot(2, 1, 2)
        plt.plot(range(len(loss_values)), loss_values, label='Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Loss Curve')
        plt.legend()

        plt.tight_layout()  # Adjusts subplot params so that the subplots fit into the figure area
        plt.draw()  # Draw the plot
        plt.pause(0.001) # Pause for a short period to allow the plot to update

        # After the training loop, you can turn off the interactive mode
        plt.ioff()
