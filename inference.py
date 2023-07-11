import torch
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from trail import test_ibll  # Assuming this is your test dataset
import matplotlib.pyplot as plt
import torchvision.transforms as T
import numpy as np
from sklearn.metrics import precision_recall_curve, PrecisionRecallDisplay
from torchvision.ops import box_iou
import numpy as np

def get_model(num_classes):
    model = fasterrcnn_resnet50_fpn(pretrained=True)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model

def collate_fn(batch):
    return tuple(zip(*batch))

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
        labels = torch.ones((len(boxes),), dtype=torch.int64)  # Assuming all objects belong to a single class
        target = {"boxes": boxes, "labels": labels}

        return image, target

    def __len__(self):
        return len(self.images)
    
def plot_image(img_tensor, annotation):
    fig, ax = plt.subplots(1)
    img = img_tensor.cpu().data

    # Convert tensor image to PIL Image and then to NumPy array
    img = T.ToPILImage()(img)
    img = np.array(img)
    
    # Display the image
    ax.imshow(img)

    # Define a threshold for the scores
    threshold = 0.1  # You can adjust this value as needed

    # Draw bounding boxes and labels
    for box, score in zip(annotation["boxes"], annotation["scores"]):
        # Move the boxes to CPU
        box = box.cpu().detach().numpy()
        xmin, ymin, xmax, ymax = box

        # Move the score to CPU and convert to numpy
        score = score.cpu().detach().numpy()

        # Only draw the boxes that have a score above the threshold
        if score > threshold:
            rect = plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, fill=False, color="red")
            ax.add_patch(rect)

            # Annotate the score of the box
            ax.text(xmin, ymax + 5, "{:.2f}".format(score), color="white", backgroundcolor="red",fontsize=5)
    
    plt.show()

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

    return precision, recall, scores


if __name__ == "__main__":
    num_classes = 2
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print("CUDA available: ", torch.cuda.is_available())

    # Load the saved model
    model_path = "./models/model_aug_epoch_24.pth"  # Change this to the path of your saved model
    model = get_model(num_classes)
    model.load_state_dict(torch.load(model_path))
    model.to(device)

    # Set the model in evaluation mode
    model.eval()

    # Load the test data
    test_targets = [data['bboxes'] for data in test_ibll.data]
    dataset = CustomDataset(test_ibll.data, test_targets)
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=2, shuffle=False, num_workers=4, collate_fn=collate_fn)

    # Perform inference
    all_targets = []
    all_predictions = []
    with torch.no_grad():
        for images, targets in data_loader:
            images = list(image.to(device) for image in images)

            predictions = model(images)
            print(predictions)
            all_targets.extend(targets)
            all_predictions.extend(predictions)

            # Plot the first image and its predicted boxes
            plot_image(images[0], predictions[0])

    precision, recall, scores = get_precision_recall(all_targets, all_predictions)

    # Plot precision-recall curve
    disp = PrecisionRecallDisplay(precision=precision, recall=recall)
    disp.plot()
            # Now "predictions" contains the output of the model. 
            # You can post-process this output as per your requirements.
