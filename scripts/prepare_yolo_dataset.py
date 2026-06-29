import os
import cv2
import shutil
import numpy as np

def convert_mask_to_yolo(mask_path):
    """
    Reads a binary mask and returns YOLO format bounding boxes:
    List of [class_id, x_center, y_center, width, height] (normalized)
    """
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        return []
    
    height, width = mask.shape
    # Threshold mask to binary (cracks are white/255, background is 0)
    _, thresh = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    
    # Find contours of the cracks
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    yolo_boxes = []
    for contour in contours:
        # Ignore extremely small noise contours (less than 10 pixels area)
        if cv2.contourArea(contour) < 10:
            continue
            
        x, y, w, h = cv2.boundingRect(contour)
        
        # Calculate normalized YOLO values
        x_center = (x + w / 2.0) / width
        y_center = (y + h / 2.0) / height
        w_norm = w / float(width)
        h_norm = h / float(height)
        
        # Class index 0 is 'crack'
        yolo_boxes.append([0, x_center, y_center, w_norm, h_norm])
        
    return yolo_boxes

def process_dataset_split(img_dir, lab_dir, dest_img_dir, dest_lab_dir):
    os.makedirs(dest_img_dir, exist_ok=True)
    os.makedirs(dest_lab_dir, exist_ok=True)
    
    files = [f for f in os.listdir(img_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    print(f"Processing {len(files)} files from {img_dir}...")
    
    count = 0
    for file_name in files:
        base_name, _ = os.path.splitext(file_name)
        mask_name = base_name + ".png" # DeepCrack labels are PNGs
        
        img_path = os.path.join(img_dir, file_name)
        mask_path = os.path.join(lab_dir, mask_name)
        
        if not os.path.exists(mask_path):
            # Try same extension just in case
            mask_name_alt = base_name + os.path.splitext(file_name)[1]
            mask_path = os.path.join(lab_dir, mask_name_alt)
            if not os.path.exists(mask_path):
                continue
        
        # 1. Copy image to YOLO directory
        dest_img_path = os.path.join(dest_img_dir, file_name)
        shutil.copy(img_path, dest_img_path)
        
        # 2. Convert mask to YOLO bounding boxes
        boxes = convert_mask_to_yolo(mask_path)
        
        # 3. Save labels file
        label_file_path = os.path.join(dest_lab_dir, base_name + ".txt")
        with open(label_file_path, "w") as lf:
            for box in boxes:
                lf.write(f"{box[0]} {box[1]:.6f} {box[2]:.6f} {box[3]:.6f} {box[4]:.6f}\n")
                
        count += 1
        
    print(f"Successfully processed {count} files.")

if __name__ == "__main__":
    # Base paths
    src_base = os.path.join("data", "segmentation", "deepcrack")
    dest_base = os.path.join("data", "yolo_dataset")
    
    # Process Train Split
    train_img = os.path.join(src_base, "train_img")
    train_lab = os.path.join(src_base, "train_lab")
    dest_train_img = os.path.join(dest_base, "train", "images")
    dest_train_lab = os.path.join(dest_base, "train", "labels")
    process_dataset_split(train_img, train_lab, dest_train_img, dest_train_lab)
    
    # Process Validation/Test Split
    test_img = os.path.join(src_base, "test_img")
    test_lab = os.path.join(src_base, "test_lab")
    dest_val_img = os.path.join(dest_base, "valid", "images")
    dest_val_lab = os.path.join(dest_base, "valid", "labels")
    process_dataset_split(test_img, test_lab, dest_val_img, dest_val_lab)
    
    print("YOLO dataset preparation complete!")
