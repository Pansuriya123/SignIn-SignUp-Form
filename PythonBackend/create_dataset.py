import os
import pickle
import mediapipe as mp
import cv2
import numpy as np

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Initialize MediaPipe Hands
hands = mp_hands.Hands(
    static_image_mode=True,
    max_num_hands=1,
    min_detection_confidence=0.5
)

DATA_DIR = './asl_dataset'  # Updated to use the correct dataset directory

data = []
labels = []
processed_count = 0
total_files = sum([len(files) for r, d, files in os.walk(DATA_DIR)])

print(f"Starting to process {total_files} images...")

for dir_ in sorted(os.listdir(DATA_DIR)):
    dir_path = os.path.join(DATA_DIR, dir_)
    if not os.path.isdir(dir_path):
        continue
        
    label_count = 0
    print(f"\nProcessing directory: {dir_}")
    
    for img_path in os.listdir(dir_path):
        if not img_path.lower().endswith(('.png', '.jpg', '.jpeg')):
            continue
            
        data_aux = []
        x_ = []
        y_ = []
        
        img = cv2.imread(os.path.join(dir_path, img_path))
        if img is None:
            print(f"Warning: Could not read image {img_path}")
            continue
            
        # Process with MediaPipe
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = hands.process(img_rgb)
        
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Draw landmarks for debugging
                debug_img = img_rgb.copy()
                mp_drawing.draw_landmarks(
                    debug_img,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style()
                )
                
                # Save debug image occasionally
                if processed_count % 100 == 0:
                    debug_path = os.path.join('debug_images', f"{dir_}_{img_path}")
                    os.makedirs('debug_images', exist_ok=True)
                    cv2.imwrite(debug_path, cv2.cvtColor(debug_img, cv2.COLOR_RGB2BGR))
                
                # Collect landmark coordinates
                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y
                    x_.append(x)
                    y_.append(y)

                # Normalize coordinates relative to the first landmark
                base_x = hand_landmarks.landmark[0].x
                base_y = hand_landmarks.landmark[0].y
                
                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y
                    # Append both normalized x and y coordinates
                    data_aux.append(x - base_x)
                    data_aux.append(y - base_y)

            data.append(data_aux)
            labels.append(dir_)
            label_count += 1
        else:
            if processed_count % 50 == 0:
                print(f"No hand detected in {img_path}")
                # Save failed detection image for debugging
                debug_path = os.path.join('failed_detections', f"{dir_}_{img_path}")
                os.makedirs('failed_detections', exist_ok=True)
                cv2.imwrite(debug_path, img)
        
        processed_count += 1
        if processed_count % 100 == 0:
            print(f"Processed {processed_count}/{total_files} images...")
            print(f"Current total samples: {len(data)}")
    
    print(f"Added {label_count} samples for label {dir_}")

print("\nData collection completed!")
print(f"Total samples collected: {len(data)}")
print("\nLabel distribution:")
for label in sorted(set(labels)):
    count = labels.count(label)
    print(f"Label {label}: {count} samples")

if len(data) == 0:
    print("\nError: No hand landmarks were detected in any images!")
    exit(1)

print("\nSaving data to data.pickle...")
f = open('data.pickle', 'wb')
pickle.dump({'data': data, 'labels': labels}, f)
f.close()
print("Data saved successfully!")   
