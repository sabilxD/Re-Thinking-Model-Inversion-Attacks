import os
import cv2
import numpy as np
from glob import glob
from tqdm import tqdm

GRID_SIZE = 8
IMAGE_SIZE = 64
FACES_PER_GRID = 60
HIST_BINS = 32

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def crop_faces_from_grid(image_path):
    img = cv2.imread(image_path)
    img = cv2.resize(img, (GRID_SIZE * IMAGE_SIZE, GRID_SIZE * IMAGE_SIZE))
    faces = []

    for i in range(GRID_SIZE):
        for j in range(GRID_SIZE):
            idx = i * GRID_SIZE + j
            if idx >= FACES_PER_GRID:
                break
            face = img[i*IMAGE_SIZE:(i+1)*IMAGE_SIZE, j*IMAGE_SIZE:(j+1)*IMAGE_SIZE]
            faces.append(face)
    return faces

def get_latest_image_for_batch(generated_dir, batch_num):
    pattern = os.path.join(generated_dir, f"{batch_num}_2400.png")
    images = sorted(glob(pattern))
    return images[-1] if images else None

def compute_histogram(image):
    hist = []
    for i in range(3):  # BGR channels
        h = cv2.calcHist([image], [i], None, [HIST_BINS], [0, 256])
        h = cv2.normalize(h, h).flatten()
        hist.append(h)
    return np.concatenate(hist)

def precompute_celeba_histograms(celeba_dir):
    filenames = sorted(os.listdir(celeba_dir))
    celeba_hists = []
    celeba_paths = []

    for fname in tqdm(filenames, desc="Computing CelebA histograms"):
        path = os.path.join(celeba_dir, fname)
        img = cv2.imread(path)
        if img is None:
            continue
        img = cv2.resize(img, (64, 64))
        hist = compute_histogram(img)
        celeba_hists.append(hist)
        celeba_paths.append(path)

    return np.array(celeba_hists), celeba_paths

def find_best_match_fast(target_img, celeba_hists, celeba_paths):
    target_hist = compute_histogram(target_img)
    # Efficient L2 distance with broadcasting
    dists = np.linalg.norm(celeba_hists - target_hist, axis=1)
    best_idx = np.argmin(dists)
    return cv2.imread(celeba_paths[best_idx]), os.path.basename(celeba_paths[best_idx])

def crop_and_match_all(generated_dir, output_root, celeba_dir):
    celeba_hists, celeba_paths = precompute_celeba_histograms(celeba_dir)

    for batch_num in range(5):
        print(f"\nProcessing batch {batch_num}")
        latest_img_path = get_latest_image_for_batch(generated_dir, batch_num)
        if latest_img_path is None:
            print(f"No images found for batch {batch_num}")
            continue

        print(f"Using image: {latest_img_path}")
        faces = crop_faces_from_grid(latest_img_path)

        batch_dir = os.path.join(output_root, f"batch_{batch_num}")
        ensure_dir(batch_dir)

        for idx, face in enumerate(faces):
            face_path = os.path.join(batch_dir, f"gen_{idx}.png")
            cv2.imwrite(face_path, face)

            best_match_img, celeb_name = find_best_match_fast(face, celeba_hists, celeba_paths)
            if best_match_img is not None:
                best_path = os.path.join(batch_dir, f"best_{idx}.png")
                cv2.imwrite(best_path, best_match_img)
                print(f"Saved best match for gen_{idx}.png as best_{idx}.png (from {celeb_name})")

if __name__ == "__main__":
    generated_dir = "attack_results/kedmi_300ids/celeba_VGG16/L_logit/imgs_L_logit"
    output_root = "cropped_faces_hist_fast"
    celeba_dir = "datasets/celeba/img_align_celeba"

    crop_and_match_all(generated_dir, output_root, celeba_dir)
