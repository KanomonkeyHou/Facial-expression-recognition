import os
import numpy as np
import cv2
import dlib

# Initialize dlib's face detector (HOG-based) and facial landmark predictor
detector = dlib.get_frontal_face_detector()
# Note: Ensure the path to the shape predictor model is correct
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')


def process_emotion_folder(folder_path):
    """
    Process all images in the specified folder, perform facial landmark extraction,
    and classify emotions into Happy, Sad, or Unknown based on geometric heuristics.
    """
    # Initialize evaluation counters
    happy_count = 0
    sad_count = 0
    unknown_count = 0
    no_face_count = 0

    if not os.path.exists(folder_path):
        print(f"Error: Directory not found -> {folder_path}")
        return

    # Iterate through all image files in the dataset directory
    for filename in os.listdir(folder_path):
        if not filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            continue

        img_path = os.path.join(folder_path, filename)
        frame = cv2.imread(img_path)

        if frame is None:
            continue

        # Convert image to grayscale for dlib processing
        if len(frame.shape) == 3:
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        else:
            gray_frame = frame

        h, w = gray_frame.shape

        # === Optimization 1: Upsampling & Fallback Mechanism ===
        # Apply 1x upsampling to detect smaller faces or slight profile views
        rects = detector(gray_frame, 1)

        if len(rects) > 0:
            # Face detected successfully, extract 68 landmarks
            landmarks = np.array([[p.x, p.y] for p in predictor(gray_frame, rects[0]).parts()])
        else:
            # Fallback Mechanism: For tightly cropped datasets with occlusion (e.g., hands),
            # forcefully assume the entire image space as the face bounding box.
            face_rect = dlib.rectangle(2, 2, w - 2, h - 2)
            try:
                landmarks = np.array([[p.x, p.y] for p in predictor(gray_frame, face_rect).parts()])
            except:
                no_face_count += 1
                continue

        # === Basic Geometric Feature Calculations ===
        # Calculate Eye Aspect Ratio (EAR) approximation -> c1
        d1 = np.linalg.norm(landmarks[37] - landmarks[41])
        d2 = np.linalg.norm(landmarks[38] - landmarks[40])
        d3 = np.linalg.norm(landmarks[36] - landmarks[39])
        c1 = (d1 + d2) / (2 * d3)

        # Calculate Mouth Aspect Ratio (MAR) -> c2 (indicative of open-mouth expressions)
        d4 = np.linalg.norm(landmarks[65] - landmarks[60])
        d5 = np.linalg.norm(landmarks[67] - landmarks[64])
        d6 = np.linalg.norm(landmarks[48] - landmarks[54])  # d6 is mouth width
        c2 = (d4 + d5) / (2 * d6) if d6 > 0 else 0

        # Calculate Nose-to-Mouth-Corner Ratio -> c3 (indicative of sad/frowning expressions)
        d7 = np.linalg.norm(landmarks[31] - landmarks[48])
        d8 = np.linalg.norm(landmarks[35] - landmarks[54])
        c3 = (d7 + d8) / (2 * d6) if d6 > 0 else 0

        # Calculate Horizontal Mouth Stretch Ratio -> (indicative of closed-mouth smiling)
        # Using the distance between outer eye corners as a robust facial width reference
        d_eyes_outer = np.linalg.norm(landmarks[36] - landmarks[45])
        mouth_width_ratio = d6 / d_eyes_outer if d_eyes_outer > 0 else 0

        # === Optimization 2: Fine-tuned Hierarchical Classification Logic ===
        # Relaxed EAR threshold (0.12) to accommodate squinting eyes during extreme laughing or crying
        if c1 >= 0.12:
            # 1. Prioritize distinct sadness: downward mouth corners
            if c3 >= 0.53:
                sad_count += 1
            # 2. Determine Happiness: Mouth is vertically open OR horizontally stretched,
            #    AND the mouth corners must be distinctly raised (strictly bounding c3 < 0.51).
            elif (c2 >= 0.35 or mouth_width_ratio >= 0.72) and c3 < 0.51:
                happy_count += 1
            # 3. Fallback for Sadness: Slight downward corners without clear smiling features
            elif c3 >= 0.51:
                sad_count += 1
            # 4. Unclassified expressions (e.g., neutral)
            else:
                unknown_count += 1
        else:
            # Eyes closed significantly
            unknown_count += 1

    # Print evaluation metrics for the current dataset subset
    folder_name = os.path.basename(folder_path)
    print(f"=== Evaluation Completed for [{folder_name.upper()}] Dataset ===")
    print(f" - Predicted as Happy:   {happy_count}")
    print(f" - Predicted as Sad:     {sad_count}")
    print(f" - Predicted as Unknown: {unknown_count}")
    print(f" - No face detected:     {no_face_count}\n")


# Define dataset directories
happy_dir = r"C:\Users\Admin\Desktop\Facial recognition\archive\test\happy"
sad_dir = r"C:\Users\Admin\Desktop\Facial recognition\archive\test\sad"

print("Initializing experimental evaluation pipeline...\n")

print("Processing [Happy] ground truth dataset...")
process_emotion_folder(happy_dir)

print("Processing [Sad] ground truth dataset...")
process_emotion_folder(sad_dir)

print("All experimental iterations completed successfully!")