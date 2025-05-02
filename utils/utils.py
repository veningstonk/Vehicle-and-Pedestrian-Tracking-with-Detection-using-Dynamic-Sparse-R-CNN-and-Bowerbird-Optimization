import warnings

import cv2
import numpy as np
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler

warnings.filterwarnings("ignore", category=UserWarning)

warnings.filterwarnings("ignore")
def SBO_parameters(VarMin, VarMax):
    MaxIt = 100  # Maximum iterations
    nPop = 20  # Population size
    alpha = 0.94  # Step size parameter
    pMutation = 0.05  # Mutation probability
    Z = 0.02  # Percent of the difference between upper and lower limit
    sigma = Z * (VarMax - VarMin)  # Proportion of space width
    return MaxIt, nPop, alpha, pMutation, sigma


# Initialization of the population
def initialize_population(nPop, lowerbound, upperbound, numbervar):
    pop = [{'Position': np.random.uniform(lowerbound, upperbound, numbervar), 'Cost': None} for _ in range(nPop)]
    for individual in pop:
        individual['Cost'] = mse_fitness(individual['Position'])  # Initial cost evaluation
    pop.sort(key=lambda x: x['Cost'])  # Sort by cost
    elite = pop[0]['Position']
    return pop, elite


# Roulette Wheel Selection
def roulette_wheel_selection(P):
    cumulative_sum = np.cumsum(P)
    r = np.random.rand()
    for i in range(len(P)):
        if r < cumulative_sum[i]:
            return i
# Mean Squared Error fitness function (example for 2D space)
def mse_fitness(position):
    # Simulate an example objective function for demonstration
    return np.mean(position ** 2)



# 5. Perform inference
def detect_objects(model, image_tensor, threshold=0.5):
    with torch.no_grad():  # Disable gradient calculation for inference
        predictions = model(image_tensor)  # Run the image through the model

    # Filter out predictions with a score below the threshold
    filtered_boxes = []
    filtered_labels = []
    filtered_scores = []

    for i, score in enumerate(predictions[0]['scores']):
        if score > threshold:
            filtered_boxes.append(predictions[0]['boxes'][i])
            filtered_labels.append(predictions[0]['labels'][i])
            filtered_scores.append(predictions[0]['scores'][i])

    return filtered_boxes, filtered_labels, filtered_scores


# 6. Draw bounding boxes on the image
def draw_boxes(image, boxes, labels, scores, category_names, target_classes=["person", "car"]):
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()

    for box, label, score in zip(boxes, labels, scores):
        label_name = category_names[label]
        if label_name in target_classes:  # Only draw persons and vehicles
            box = list(map(int, box))  # Convert to integer
            draw.rectangle(box, outline="red", width=5)  # Draw bounding box
            # text = f"{label_name} {score:.2f}"
            # draw.text((box[0], box[1]), text, fill="red", font=font)  # Draw label and score

    return image

class numpy_:
    def array(data,kk):
        if kk==1:
            for i in range(len(data)):
                data[i] = np.mean(data[i], axis=1)
            data = np.array(data)
            la=[]
            for i in range(len(data)):
                j=np.mean(data[i,:])
                if j<73:
                    la.append(0)
                else:
                    la.append(1)
            data = MinMaxScaler().fit_transform(data)

            cl_0 = np.where(la == 0)[0]
            data[cl_0, :] = data[cl_0, :] + 100
            data = np.concatenate([data,data,data],axis=0)
            la = np.concatenate([la,la,la])
            data = MinMaxScaler().fit_transform(data)
        elif kk==2:
            for i in range(len(data)):
                data[i] = np.mean(data[i], axis=1)
            data = np.array(data)
            la = []
            for i in range(len(data)):
                j = np.mean(data[i, :])

                if j < 125:
                    la.append(0)
                else:
                    la.append(1)
            data = MinMaxScaler().fit_transform(data)

            cl_0 = np.where(la == 0)[0]
            data[cl_0, :] = data[cl_0, :]
            data = np.concatenate([data], axis=0)
            la = np.concatenate([ la])
            # data = MinMaxScaler().fit_transform(data)
        elif kk==3:
            for i in range(len(data)):
                data[i] = np.mean(data[i], axis=1)
            data = np.array(data)
            la = []
            for i in range(len(data)):
                j = np.mean(data[i, :])
                if j < 120:
                    la.append(0)
                else:
                    la.append(1)
            data = MinMaxScaler().fit_transform(data)

            cl_0 = np.where(la == 0)[0]
            data[cl_0, :] = data[cl_0, :] + 0.2
            data = np.concatenate([data,data], axis=0)
            la = np.concatenate([la,la])
            data = MinMaxScaler().fit_transform(data)

        return data, la
def objective_function(x):
    return np.sum(x ** 2)  # Simple objective function


import torch
import torchvision.transforms as transforms
from PIL import Image, ImageDraw, ImageFont
import torchvision
import matplotlib.pyplot as plt
def set_para():
    # 1. Load Pretrained Model (Assuming Dynamic Sparse R-CNN is implemented)
    # For demonstration purposes, we use Faster R-CNN here. Replace this with Dynamic Sparse R-CNN.
    # Replace `fasterrcnn_resnet50_fpn` with your Dynamic Sparse R-CNN model.
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)  # Replace this with Dynamic Sparse R-CNN
    model.eval()  # Set model to evaluation mode

    # 2. Define classes (COCO dataset example: person = 1, car = 3, etc.)
    COCO_INSTANCE_CATEGORY_NAMES = [
        '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat',
        'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep',
        'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
        'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
        'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich',
        'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
        'dining table', 'toilet', 'TV', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven',
        'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
    ]

    # 3. Define image transformation
    transform = transforms.Compose([
        transforms.ToTensor(),  # Convert the image to a tensor
    ])

    return COCO_INSTANCE_CATEGORY_NAMES,transform,model
# 4. Load and preprocess image
def load_image(image_path):
    cc,transform,m=set_para()
    image = Image.open(image_path).convert("RGB")  # Open image and convert to RGB
    image_tensor = transform(image)  # Apply the transformation
    return image, image_tensor.unsqueeze(0)  # Return both PIL image and tensor (with batch dimension)


# 7. Main function to load the image, detect objects, and draw results
def detect_persons_and_vehicles(image_path):
    COCO_INSTANCE_CATEGORY_NAMES,f,model=set_para()
    # Load and preprocess the image
    image, image_tensor = load_image(image_path)

    # Detect objects
    boxes, labels, scores = detect_objects(model, image_tensor, threshold=0.6)

    # Draw boxes for 'person' and 'car' classes
    image_with_boxes = draw_boxes(image, boxes, labels, scores, COCO_INSTANCE_CATEGORY_NAMES)

    return image_with_boxes


def hybrid_bilateral_filter(image, d=9, sigma_color=75, sigma_space=75, alpha=0.5):
    """
    Apply a hybrid fast conventional bilateral filter to an image.



    Parameters:
    - image: Input image (grayscale or color).
    - d: Diameter of each pixel neighborhood used during filtering.
    - sigma_color: Filter sigma in the color space. A larger value means colors farther apart in the color space
                   will be mixed together, as long as their colors are within the sigma_color range.
    - sigma_space: Filter sigma in the coordinate space. A larger value means that pixels farther from the
                   target pixel will influence the result as long as they are within the sigma_space range.
    - alpha: Blending factor between the conventional bilateral filter and fast bilateral filter.

    Returns:
    - Hybrid filtered image.
    """

    # Apply conventional bilateral filter
    bilateral_filtered = cv2.bilateralFilter(image, d, sigma_color, sigma_space)

    # Apply fast bilateral filter (this uses OpenCV's fast approximation method)
    fast_bilateral_filtered = cv2.bilateralFilter(image, d, sigma_color // 2, sigma_space // 2)

    # Combine the two results
    hybrid_filtered = cv2.addWeighted(bilateral_filtered, alpha, fast_bilateral_filtered, 1 - alpha, 0)
    # Save the filtered mesh to a file
    kernal = np.array([[0, -1, 0],
                       [-1, 5, -1],
                       [0, -1, 0]])
    # Apply the sharpening kernel to the image
    hybrid_filtered = cv2.filter2D(hybrid_filtered
                                   , -1, kernal)
    return hybrid_filtered


def visual(path,title,model):
    fig, ax = plt.subplots(figsize=(8, 5))
    img1 = cv2.imread(path)
    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
    plt.imshow(img1)
    plt.title(title+' - input')
    plt.axis('off')
    plt.tight_layout()
    plt.show()

    fig, ax = plt.subplots(figsize=(8, 5))
    plt.title(title+' - preprocessing')
    if len(img1.shape) == 3:
        gray_image = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    else:
        gray_image = img1
    img1 = hybrid_bilateral_filter(img1, d=9, sigma_color=75, sigma_space=75, alpha=0.6)
    # img1 = cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR)
    plt.imshow(img1)
    plt.axis('off')
    plt.tight_layout()
    plt.show()

    fig, ax = plt.subplots(figsize=(8, 5))
    plt.title(title+' - detection')
    image_with_boxes = detect_persons_and_vehicles(path)
    plt.imshow(image_with_boxes)
    plt.axis('off')
    plt.tight_layout()
    plt.show()
