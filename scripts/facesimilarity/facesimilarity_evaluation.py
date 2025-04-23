import os
import random
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm
from sklearn.metrics import roc_curve, auc, f1_score, confusion_matrix
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import cosine_similarity
import torch
from facenet_pytorch import MTCNN, InceptionResnetV1
import cv2

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

def extract_embedding(img_path):
    img = Image.open(img_path)
    face = mtcnn(img)
    if face is None:
        return None
    emb = model(face.unsqueeze(0).to(device))
    return emb.detach().cpu().numpy()[0]

def load_embeddings_from_dataset(dataset_dir):
    embeddings = []
    labels = []
    image_paths = []

    for class_name in os.listdir(dataset_dir):
        class_dir = os.path.join(dataset_dir, class_name)
        if not os.path.isdir(class_dir):
            continue
        for img_name in os.listdir(class_dir):
            img_path = os.path.join(class_dir, img_name)
            emb = extract_embedding(img_path)
            if emb is not None:
                embeddings.append(emb)
                labels.append(class_name)
                image_paths.append(img_path)
    
    return np.array(embeddings), np.array(labels), image_paths

def plot_tsne(embeddings, labels):
    print("[INFO] Menghitung t-SNE...")
    tsne = TSNE(n_components=2, random_state=42)
    emb_2d = tsne.fit_transform(embeddings)
    plt.figure(figsize=(8, 6))
    for label in np.unique(labels):
        idx = labels == label
        plt.scatter(emb_2d[idx, 0], emb_2d[idx, 1], label=label, alpha=0.6)
    plt.legend()
    plt.title("Visualisasi t-SNE dari Face Embeddings")
    plt.show()

def generate_pairs(embeddings, labels, image_paths, max_pairs=1000):
    positive_pairs = []
    negative_pairs = []
    index_pairs = []

    label_to_indices = {label: np.where(labels == label)[0] for label in np.unique(labels)}

    for label, indices in label_to_indices.items():
        if len(indices) < 2:
            continue
        for _ in range(min(max_pairs, len(indices) * 2)):
            i, j = np.random.choice(indices, 2, replace=False)
            positive_pairs.append((embeddings[i], embeddings[j], 1))
            index_pairs.append((i, j))

    all_indices = list(range(len(labels)))
    for _ in range(len(positive_pairs)):
        while True:
            i, j = np.random.choice(all_indices, 2, replace=False)
            if labels[i] != labels[j]:
                negative_pairs.append((embeddings[i], embeddings[j], 0))
                index_pairs.append((i, j))
                break

    all_pairs = positive_pairs + negative_pairs
    return all_pairs, index_pairs

def evaluate_roc(pairs):
    similarities = []
    labels = []

    for emb1, emb2, label in pairs:
        sim = cosine_similarity([emb1], [emb2])[0][0]
        similarities.append(sim)
        labels.append(label)

    fpr, tpr, thresholds = roc_curve(labels, similarities)
    roc_auc = auc(fpr, tpr)

    optimal_idx = np.argmax(tpr - fpr)
    optimal_threshold = thresholds[optimal_idx]

    predictions = [1 if sim >= optimal_threshold else 0 for sim in similarities]
    cm = confusion_matrix(labels, predictions)
    tn, fp, fn, tp = cm.ravel()

    TAR = tp / (tp + fn)
    FAR = fp / (fp + tn)
    FRR = fn / (tp + fn)
    EER = fpr[np.nanargmin(np.absolute((1 - tpr) - fpr))]
    f1 = f1_score(labels, predictions)

    print(f"Optimal Threshold: {optimal_threshold:.4f}")
    print(f"TAR: {TAR:.4f} | FAR: {FAR:.4f} | FRR: {FRR:.4f} | EER: {EER:.4f} | F1-Score: {f1:.4f} | AUC: {roc_auc:.4f}")

    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    plt.grid(True)
    plt.show()

    return optimal_threshold

def plot_similarity_distribution(pairs):
    same_scores = []
    diff_scores = []

    for emb1, emb2, label in pairs:
        score = cosine_similarity([emb1], [emb2])[0][0]
        if label == 1:
            same_scores.append(score)
        else:
            diff_scores.append(score)

    plt.figure(figsize=(8, 5))
    plt.hist(same_scores, bins=50, alpha=0.6, label="Same Identity", color='green')
    plt.hist(diff_scores, bins=50, alpha=0.6, label="Different Identity", color='red')
    plt.axvline(np.mean(same_scores), color='green', linestyle='--', label="Mean Same")
    plt.axvline(np.mean(diff_scores), color='red', linestyle='--', label="Mean Different")
    plt.title("Distribusi Skor Similarity (Cosine)")
    plt.xlabel("Similarity Score")
    plt.ylabel("Jumlah Pasangan")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def save_example_visuals(pairs_with_paths, threshold, save_dir="examples"):
    os.makedirs(save_dir, exist_ok=True)
    categories = {"tp": None, "fp": None, "fn": None, "tn": None}

    for (emb1, emb2, label), (path1, path2) in pairs_with_paths:
        sim = cosine_similarity([emb1], [emb2])[0][0]
        pred = 1 if sim >= threshold else 0

        if label == 1 and pred == 1 and categories["tp"] is None:
            categories["tp"] = (path1, path2, sim)
        elif label == 0 and pred == 1 and categories["fp"] is None:
            categories["fp"] = (path1, path2, sim)
        elif label == 1 and pred == 0 and categories["fn"] is None:
            categories["fn"] = (path1, path2, sim)
        elif label == 0 and pred == 0 and categories["tn"] is None:
            categories["tn"] = (path1, path2, sim)

        if all(v is not None for v in categories.values()):
            break

    for category, data in categories.items():
        if data is None:
            continue
        path1, path2, sim = data
        img1 = cv2.cvtColor(cv2.imread(path1), cv2.COLOR_BGR2RGB)
        img2 = cv2.cvtColor(cv2.imread(path2), cv2.COLOR_BGR2RGB)
        fig, axs = plt.subplots(1, 2, figsize=(6, 3))
        axs[0].imshow(img1)
        axs[0].set_title("Gambar 1")
        axs[1].imshow(img2)
        axs[1].set_title("Gambar 2")
        for ax in axs:
            ax.axis("off")
        plt.suptitle(f"{category.upper()} - Similarity: {sim:.3f}", fontsize=12)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f"{category}.png"))
        plt.close()

if __name__ == "__main__":
    DATASET_PATH = "Dataset/Cropped"

    print("[INFO] Memuat embedding dari dataset...")
    embeddings, labels, image_paths = load_embeddings_from_dataset(DATASET_PATH)

    print("[INFO] Visualisasi embedding dengan t-SNE...")
    plot_tsne(embeddings, labels)

    print("[INFO] Evaluasi dengan ROC curve dan metrik lainnya...")
    pairs, index_pairs = generate_pairs(embeddings, labels, image_paths)
    threshold = evaluate_roc(pairs)

    print("[INFO] Visualisasi distribusi skor similarity...")
    plot_similarity_distribution(pairs)

    pairs_with_paths = []
    for (emb1, emb2, label), (idx1, idx2) in zip(pairs, index_pairs):
        path1 = image_paths[idx1]
        path2 = image_paths[idx2]
        pairs_with_paths.append(((emb1, emb2, label), (path1, path2)))

    print("[INFO] Menyimpan contoh visualisasi TP, FP, FN, TN...")
    save_example_visuals(pairs_with_paths, threshold=threshold)
