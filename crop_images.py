import os
import cv2
import numpy as np
from glob import glob
from tqdm import tqdm
import face_recognition
import pickle

GRID_SIZE = 8
IMAGE_SIZE = 64
FACES_PER_GRID = 60

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

# ----- STEP 1: Precompute CelebA Encodings -----
def precompute_celeba_encodings(celeba_dir, encodings_file):
    if os.path.exists(encodings_file):
        print(f"[INFO] Using cached CelebA encodings: {encodings_file}")
        with open(encodings_file, 'rb') as f:
            return pickle.load(f)
    else:
        print(f"[ERROR] Encodings file not found: {encodings_file}")
        exit(1)  # Or `sys.exit(1)` if you prefer



# ----- STEP 2: Crop Faces from Generated Grid -----
def crop_faces_from_grid(image_path):
    img = cv2.imread(image_path)
    img = cv2.resize(img, (GRID_SIZE * IMAGE_SIZE, GRID_SIZE * IMAGE_SIZE))
    faces = []

    for i in range(GRID_SIZE):
        for j in range(GRID_SIZE):
            idx = i * GRID_SIZE + j
            if idx >= FACES_PER_GRID:
                break
            face = img[i * IMAGE_SIZE:(i + 1) * IMAGE_SIZE, j * IMAGE_SIZE:(j + 1) * IMAGE_SIZE]
            faces.append(face)
    return faces


# ----- STEP 3: Matching Function -----
def find_best_match(target_img, celeb_encodings):
    rgb_target = cv2.cvtColor(target_img, cv2.COLOR_BGR2RGB)
    encodings = face_recognition.face_encodings(rgb_target)

    if not encodings:
        return None, None  # No face found
    target_encoding = encodings[0]

    min_distance = float('inf')
    best_match = None
    best_name = None

    for entry in celeb_encodings:
        dist = np.linalg.norm(target_encoding - entry['encoding'])
        if dist < min_distance:
            min_distance = dist
            best_match = cv2.imread(os.path.join(celeba_dir, entry['filename']))
            best_name = entry['filename']

    return best_match, best_name


# ----- STEP 4: Load Generated Image and Find Best Match -----
def crop_and_match_all(generated_dir, output_root, celeba_dir, encodings_file):
    celeba_encodings = precompute_celeba_encodings(celeba_dir, encodings_file)
    ensure_dir(output_root)  # Ensure output folder exists
    i=0
    for batch_num in range(5):
        print(f"\nProcessing batch {batch_num}")
        latest_img_path = get_latest_image_for_batch(generated_dir, batch_num)
        if latest_img_path is None:
            print(f"No images found for batch {batch_num}")
            continue

        print(f"Using image: {latest_img_path}")
        faces = crop_faces_from_grid(latest_img_path)

        for idx, face in enumerate(faces):
            face_filename = f"gen_{i}.png"
            face_path = os.path.join(output_root, face_filename)
            cv2.imwrite(face_path, face)

            best_match_img, celeb_name = find_best_match(face, celeba_encodings)
            if best_match_img is not None:
                best_filename = f"best_{i}.png"
                best_path = os.path.join(output_root, best_filename)
                cv2.imwrite(best_path, best_match_img)
                print(f"Saved best match for {face_filename} as {best_filename} (from {celeb_name})")
            i+=1



def get_latest_image_for_batch(generated_dir, batch_num):
    pattern = os.path.join(generated_dir, f"{batch_num}_2400.png")
    images = sorted(glob(pattern))
    return images[-1] if images else None


# ----- MAIN -----
if __name__ == "__main__":
    generated_dir = "attack_results/kedmi_300ids/celeba_VGG16/ours/imgs_ours"  # Folder with generated images
    output_root = "cropped_faces/Logit_loss+model_augmentation/"     # Output folder to hold batch_i/gen_*.png and best_*.png
    celeba_dir = "datasets/celeba/img_align_celeba"  # Folder with CelebA images
    encodings_file = "celeba_encodings.pkl"  # Where to save the CelebA encodings

    crop_and_match_all(generated_dir, output_root, celeba_dir, encodings_file)
