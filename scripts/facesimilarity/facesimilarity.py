import os
import cv2
import torch
import numpy as np
from facenet_pytorch import MTCNN, InceptionResnetV1
from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
mtcnn = MTCNN(image_size=160, margin=0, device=device)
# model = InceptionResnetV1(pretrained='vggface2').eval().to(device)
# print(f"[INFO] Using device: {device}")
model = InceptionResnetV1(classify=False, pretrained=None).to(device)
# model.load_state_dict(torch.load("finetuned_facenet.pth", map_location=device))
state_dict = torch.load("model/finetuned_facenet.pth", map_location=device)
filtered_state_dict = {k: v for k, v in state_dict.items() if not k.startswith("logits")}
model.load_state_dict(filtered_state_dict, strict=False)
model.eval()

def extract_embedding(image_path):
    img = Image.open(image_path)
    face = mtcnn(img)
    if face is None:
        print(f"[X] Wajah tidak terdeteksi pada {image_path}")
        return None
    face_embedding = model(face.unsqueeze(0).to(device))
    return face_embedding.detach().cpu().numpy()

def calculate_similarity(emb1, emb2):
    return cosine_similarity(emb1, emb2)[0][0]

def visualize_comparison(img_path1, img_path2, similarity_score, threshold=0.8):
    match = "MATCH" if similarity_score >= threshold else "NOT MATCH"

    img1 = cv2.imread(img_path1)
    img2 = cv2.imread(img_path2)
    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)

    fig, axs = plt.subplots(1, 2, figsize=(8, 4))
    axs[0].imshow(img1)
    axs[0].set_title("Gambar 1")
    axs[1].imshow(img2)
    axs[1].set_title("Gambar 2")

    for ax in axs:
        ax.axis('off')
    
    plt.suptitle(f"Similarity Score: {similarity_score:.3f} → {match}", fontsize=14)
    plt.tight_layout()
    plt.show()

def main():
    print("=== FACE SIMILARITY CHECK ===")
    path1 = input("Masukkan path gambar 1: ")
    path2 = input("Masukkan path gambar 2: ")

    emb1 = extract_embedding(path1)
    emb2 = extract_embedding(path2)

    if emb1 is not None and emb2 is not None:
        similarity = calculate_similarity(emb1, emb2)
        visualize_comparison(path1, path2, similarity)
    else:
        print("[!] Salah satu gambar gagal diproses.")

if __name__ == "__main__":
    main()
