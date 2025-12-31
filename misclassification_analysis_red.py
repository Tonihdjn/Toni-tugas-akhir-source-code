import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Configuration
class_names = ['gun', 'person', 'person with a gun']
class_colors = [(255, 0, 0), (255, 255, 0), (255, 0, 255)]  # Red, Yellow, Magenta
dataset_path = "Armed-Person-Recognition-4"

def parse_yolo_txt(txt_path, img_width, img_height):
    boxes = []
    if os.path.exists(txt_path):
        with open(txt_path, 'r') as f:
            for line in f.readlines():
                parts = line.strip().split()
                if len(parts) >= 5:
                    cls = int(parts[0])
                    x_center = float(parts[1]) * img_width
                    y_center = float(parts[2]) * img_height
                    width = float(parts[3]) * img_width
                    height = float(parts[4]) * img_height
                    
                    x1 = int(x_center - width/2)
                    y1 = int(y_center - height/2)
                    x2 = int(x_center + width/2)
                    y2 = int(y_center + height/2)
                    
                    confidence = float(parts[5]) if len(parts) > 5 else 1.0
                    boxes.append([cls, x1, y1, x2, y2, confidence])
    return boxes

def has_3_objects_3_classes(gt_boxes):
    if len(gt_boxes) != 3:
        return False
    classes = set([box[0] for box in gt_boxes])
    return len(classes) == 3

def calculate_iou(box1, box2):
    x1 = max(box1[1], box2[1])
    y1 = max(box1[2], box2[2])
    x2 = min(box1[3], box2[3])
    y2 = min(box1[4], box2[4])
    
    if x2 <= x1 or y2 <= y1:
        return 0.0
    
    intersection = (x2 - x1) * (y2 - y1)
    area1 = (box1[3] - box1[1]) * (box1[4] - box1[2])
    area2 = (box2[3] - box2[1]) * (box2[4] - box2[2])
    union = area1 + area2 - intersection
    
    return intersection / union if union > 0 else 0.0

def find_misclassifications(gt_boxes, pred_boxes, iou_threshold=0.5, filter_gun=True, filter_person=True, filter_person_with_gun=True):
    misclassifications = []
    
    for pred_box in pred_boxes:
        best_iou = 0
        best_gt_box = None
        
        for gt_box in gt_boxes:
            iou = calculate_iou(pred_box, gt_box)
            if iou > best_iou:
                best_iou = iou
                best_gt_box = gt_box
        
        if best_iou >= iou_threshold and best_gt_box is not None:
            if pred_box[0] != best_gt_box[0]:
                gt_class = best_gt_box[0]
                pred_class = pred_box[0]
                
                include_misclass = False
                if (gt_class == 0 or pred_class == 0) and filter_gun:
                    include_misclass = True
                if (gt_class == 1 or pred_class == 1) and filter_person:
                    include_misclass = True
                if (gt_class == 2 or pred_class == 2) and filter_person_with_gun:
                    include_misclass = True
                
                if include_misclass:
                    misclassifications.append({
                        'gt_class': gt_class,
                        'pred_class': pred_class,
                        'gt_box': best_gt_box,
                        'pred_box': pred_box,
                        'iou': best_iou
                    })
    
    return misclassifications

def draw_gt_image(img, gt_boxes):
    img_copy = img.copy()
    for box in gt_boxes:
        cls, x1, y1, x2, y2 = box[:5]
        color = class_colors[cls]
        cv2.rectangle(img_copy, (x1, y1), (x2, y2), color, 2)
        cv2.putText(img_copy, class_names[cls], (x1, y1-10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    return img_copy

def draw_pred_image(img, pred_boxes, misclassifications):
    img_copy = img.copy()
    misc_pred_boxes = [misc['pred_box'] for misc in misclassifications]
    
    for box in pred_boxes:
        cls, x1, y1, x2, y2 = box[:5]
        is_misclassified = any(np.array_equal(box[:5], misc_box[:5]) for misc_box in misc_pred_boxes)
        
        if is_misclassified:
            misc_info = next(misc for misc in misclassifications if np.array_equal(box[:5], misc['pred_box'][:5]))
            color = (255, 0, 0)  # Red for misclassified
            label = f"{class_names[cls]} ({class_names[misc_info['gt_class']]})"
        else:
            color = class_colors[cls]
            label = class_names[cls]
        
        cv2.rectangle(img_copy, (x1, y1), (x2, y2), color, 2)
        cv2.putText(img_copy, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    
    return img_copy

def analyze_model_results(model_name, filter_gun=True, filter_person=True, filter_person_with_gun=True, require_3_objects_3_classes=True):
    validate_dir = f"validate/{model_name}/val"
    labels_dir = os.path.join(validate_dir, "labels")
    
    if not os.path.exists(labels_dir):
        print(f"Labels directory not found: {labels_dir}")
        return None
    
    gt_labels_dir = os.path.join(dataset_path, "valid", "labels")
    images_dir = os.path.join(dataset_path, "valid", "images")
    all_misclassifications = []
    
    for pred_file in os.listdir(labels_dir):
        if not pred_file.endswith('.txt'):
            continue
            
        img_name = pred_file.replace('.txt', '.jpg')
        img_path = os.path.join(images_dir, img_name)
        gt_path = os.path.join(gt_labels_dir, pred_file)
        pred_path = os.path.join(labels_dir, pred_file)
        
        if not all(os.path.exists(p) for p in [img_path, gt_path]):
            continue
            
        img = cv2.imread(img_path)
        if img is None:
            continue
            
        h, w = img.shape[:2]
        gt_boxes = parse_yolo_txt(gt_path, w, h)
        pred_boxes = parse_yolo_txt(pred_path, w, h)
        
        if require_3_objects_3_classes and not has_3_objects_3_classes(gt_boxes):
            continue
        
        misclassifications = find_misclassifications(gt_boxes, pred_boxes, 0.5, filter_gun, filter_person, filter_person_with_gun)
        
        if misclassifications:
            all_misclassifications.append({
                'image': img_name,
                'image_path': img_path,
                'gt_boxes': gt_boxes,
                'pred_boxes': pred_boxes,
                'misclassifications': misclassifications
            })
    
    return all_misclassifications

def show_misclassification_examples(model_name, max_examples=6, filter_gun=True, filter_person=True, filter_person_with_gun=True, require_3_objects_3_classes=True):
    results = analyze_model_results(model_name, filter_gun, filter_person, filter_person_with_gun, require_3_objects_3_classes)
    
    if not results:
        print(f"No misclassifications found for {model_name} with current filters")
        return
    
    examples = results[:max_examples]
    rows = len(examples)
    cols = 2
    
    fig, axes = plt.subplots(rows, cols, figsize=(12, 4*rows))
    if rows == 1:
        axes = axes.reshape(1, -1)
    
    for i, example in enumerate(examples):
        img = cv2.imread(example['image_path'])
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        img_gt = draw_gt_image(img_rgb, example['gt_boxes'])
        axes[i, 0].imshow(img_gt)
        axes[i, 0].set_title(f"Ground Truth\n{example['image']}")
        axes[i, 0].axis('off')
        
        img_pred = draw_pred_image(img_rgb, example['pred_boxes'], example['misclassifications'])
        axes[i, 1].imshow(img_pred)
        axes[i, 1].set_title(f"Predictions\n{len(example['misclassifications'])} misclassifications")
        axes[i, 1].axis('off')
    
    plt.tight_layout()
    plt.show()
    
    return results