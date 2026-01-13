import os
import torch
import lpips
from PIL import Image
import torchvision.transforms as transforms
from tqdm import tqdm
import csv
import json
import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

LPIPS_THRESHOLD = 0.3  # tweak this based on what you consider "high"
loss_fn = lpips.LPIPS(net='alex')  # can be 'alex', 'squeeze', or 'vgg'



# Preprocessing
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5],
                         std=[0.5, 0.5, 0.5])
])


def compute_lpips(img1_pil, img2_pil):
    # LPIPS model

    img1 = transform(img1_pil).unsqueeze(0)
    img2 = transform(img2_pil).unsqueeze(0)
    with torch.no_grad():
        dist = loss_fn(img1, img2)
    return dist.item()

def compute_l2_norm(img1_pil, img2_pil):
    img1 = np.array(img1_pil, dtype=np.float32)
    img2 = np.array(img2_pil, dtype=np.float32)
    l2 = np.linalg.norm(img1 - img2)
    return l2

def compute_linf_norm(img1_pil, img2_pil):
    img1 = np.array(img1_pil, dtype=np.float32)
    img2 = np.array(img2_pil, dtype=np.float32)
    linf = np.max(np.abs(img1 - img2))
    return linf

def compute_psnr(img1_pil, img2_pil):
    img1 = np.array(img1_pil, dtype=np.uint8)
    img2 = np.array(img2_pil, dtype=np.uint8)
    psnr = peak_signal_noise_ratio(img1, img2, data_range=255)
    return psnr

def compute_ssim(img1_pil, img2_pil):
    img1 = np.array(img1_pil, dtype=np.uint8)
    img2 = np.array(img2_pil, dtype=np.uint8)
    ssim = structural_similarity(img1, img2, channel_axis=2, data_range=255)
    return ssim


def calculate_naturalness(orig_root, aes_root):
    scores = []
    high_lpips = []

    l2_scores = []
    linf_scores = []
    psnr_scores = []
    ssim_scores = []

    # Traverse subfolders
    for cls in sorted(os.listdir(orig_root)):
        orig_cls_dir = os.path.join(orig_root, cls)
        aes_cls_dir  = os.path.join(aes_root, cls)
        if not os.path.isdir(orig_cls_dir) or not os.path.isdir(aes_cls_dir):
            continue

        orig_files = os.listdir(orig_cls_dir)
        aes_files  = os.listdir(aes_cls_dir)

        orig_dict = {f"{f.split('_')[0]}-{f.split('_')[1]}": f for f in orig_files}
        aes_dict  = {f"{f.split('_')[0]}-{f.split('_')[1]}": f for f in aes_files}

        common_ids = set(orig_dict.keys()) & set(aes_dict.keys())

        for file_id in tqdm(common_ids, desc=f"Class {cls}"):
            orig_path = os.path.join(orig_cls_dir, orig_dict[file_id])
            aes_path  = os.path.join(aes_cls_dir, aes_dict[file_id])

            # Load PIL Images, resize ONCE here
            orig_pil = Image.open(orig_path).convert("RGB").resize((224,224))
            aes_pil  = Image.open(aes_path).convert("RGB").resize((224,224))

            score = compute_lpips(orig_pil, aes_pil)
            scores.append(score)

            # Added: Compute extra metrics
            l2 = compute_l2_norm(orig_pil, aes_pil)
            linf = compute_linf_norm(orig_pil, aes_pil)
            psnr = compute_psnr(orig_pil, aes_pil)
            ssim = compute_ssim(orig_pil, aes_pil)

            l2_scores.append(l2)
            linf_scores.append(linf)
            psnr_scores.append(psnr)
            ssim_scores.append(ssim)

            # Log high LPIPS samples (also log all metrics for them)
            if score > LPIPS_THRESHOLD:
                high_lpips.append((cls, file_id, orig_path, aes_path, score, l2, psnr, ssim))

   
    with open(f"{os.path.dirname(orig_root)}/high_lpips_samples.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["class", "id", "orig_path", "aes_path", "lpips_score", "l2_norm", "psnr", "ssim"])
        writer.writerows(high_lpips)
        
    result = {
     
        "mean_lpips": round(sum(scores)/len(scores), 4),
        f"high_lips_count(>{LPIPS_THRESHOLD})": len(high_lpips),
        "mean_l2": round(sum(l2_scores)/len(l2_scores), 4),
        "mean_linf": round(sum(linf_scores)/len(linf_scores), 4),
        "mean_psnr": round(sum(psnr_scores)/len(psnr_scores), 4),
        "mean_ssim": round(sum(ssim_scores)/len(ssim_scores), 4)
    
    }
    return result

    # print(json.dumps(result, indent=4))

