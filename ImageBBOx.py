import os
import xml.etree.ElementTree as ET
from collections import Counter
import matplotlib.pyplot as plt
import random
import cv2
import imgaug.augmenters as iaa
from imgaug.augmentables.bbs import BoundingBox, BoundingBoxesOnImage

class ImageBBoxLabelList:
    def __init__(self, path):
        self.path = path
        self.data = []
    
    @classmethod
    def from_pascal_voc(cls, path):
        self = cls(path)
        for file in os.listdir(path):
            if file.endswith(".xml"):
                file_path = os.path.join(path, file)
                tree = ET.parse(file_path)
                root = tree.getroot()
                
                bboxes = []
                labels = []
                for obj in root.findall('object'):
                    bbox = obj.find('bndbox')
                    xmin = int(bbox.find('xmin').text)
                    ymin = int(bbox.find('ymin').text)
                    xmax = int(bbox.find('xmax').text)
                    ymax = int(bbox.find('ymax').text)
                    bboxes.append((xmin, ymin, xmax, ymax))
                    
                    label = obj.find('name').text
                    labels.append(label)

                # Assuming images are in the same directory and have the same name as the XML files
                img_file = file_path.replace('.xml', '.jpg')  # Replace with your image file extension
                img = cv2.imread(img_file)

                self.data.append({
                    'filename': file,
                    'bboxes': bboxes,
                    'labels': labels,
                    'img': img  # Added image here
                })
        return self

    def show_dist(self, train_data=None, test_data=None):
        # Count all labels
        label_counter = Counter()
        for item in self.data:
            label_counter.update(item['labels'])

        # Add background class with count 0
        label_counter['background'] = 0

        # Prepare data for the bar chart
        labels = list(label_counter.keys())
        counts = list(label_counter.values())

        # Plot the bar chart
        plt.bar(labels, counts, color='b', label='Total')
        
        if train_data is not None:
            train_label_counter = Counter()
            for item in train_data.data:
                train_label_counter.update(item['labels'])
            train_counts = [train_label_counter[label] for label in labels]
            plt.bar(labels, train_counts, color='c', label='Train')
        
        if test_data is not None:
            test_label_counter = Counter()
            for item in test_data.data:
                test_label_counter.update(item['labels'])
            test_counts = [test_label_counter[label] for label in labels]
            plt.bar(labels, test_counts, color='b', alpha=0.5, label='Test')

        plt.title('Distribution of classes')
        plt.xlabel('Classes')
        plt.ylabel('Count')
        plt.legend()
        plt.show()


    @staticmethod
    def merge(ibll1, ibll2):
        merged_data = ibll1.data + ibll2.data
        merged_ibll = ImageBBoxLabelList('merged_path')
        merged_ibll.data = merged_data
        return merged_ibll

    def split(self, train_ratio=0.8, show=False):
        # Calculate the number of samples for train and test based on the train_ratio
        num_samples = len(self.data)
        num_train = int(num_samples * train_ratio)
        num_test = num_samples - num_train

        # Shuffle the data randomly
        random.shuffle(self.data)

        # Split the data into train and test sets
        train_data = self.data[:num_train]
        test_data = self.data[num_train:]

        # Create new instances of ImageBBoxLabelList for train and test sets
        train_ibll = ImageBBoxLabelList('train_path')
        train_ibll.data = train_data

        test_ibll = ImageBBoxLabelList('test_path')
        test_ibll.data = test_data

        # Show the distribution if requested
        if show:
            self.show_dist(train_ibll, test_ibll)

        # Return the train and test instances
        return train_ibll, test_ibll

    def set_tfms(self, tfms):

        self.tfms = tfms

    def apply_tfms(self):
        """
        Apply the transformations to the images in the dataset.
        """
        # Iterate over all data entries
        for i in range(len(self.data)):
            # Extract the image and bounding boxes from the current data entry
            img = self.data[i]['img']
            bboxes = self.data[i]['bboxes']
            labels = self.data[i]['labels'] # Added extraction of labels here
            
            # Convert bounding boxes to BoundingBoxesOnImage object
            bboxes = [BoundingBox(*bbox) for bbox in bboxes]
            bboxes_on_img = BoundingBoxesOnImage(bboxes, shape=img.shape)
                
            # Apply all transformations
            for tfm in self.tfms:
                if isinstance(tfm, iaa.meta.Augmenter):
                    # Augment image and bounding boxes
                    img, bboxes_on_img = tfm(image=img, bounding_boxes=bboxes_on_img)
                    bboxes_on_img = bboxes_on_img.remove_out_of_image().clip_out_of_image()
                    
                    # Convert bounding boxes back to simple array format
                    bboxes = [[bbox.x1_int, bbox.y1_int, bbox.x2_int, bbox.y2_int] for bbox in bboxes_on_img.bounding_boxes]
                    
                    self.data[i]['img'] = img
                    self.data[i]['bboxes'] = bboxes
                    self.data[i]['labels'] = labels # Added this line
                else:
                    # If the transform is a function, we assume it takes a dictionary containing the image and bounding boxes as input
                    data = {'img': img, 'bboxes': bboxes, 'labels': labels}
                    data = tfm(data)
                    self.data[i] = data



