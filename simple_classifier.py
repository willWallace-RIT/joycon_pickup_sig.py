import numpy as np
import os
from collections import defaultdict

DATA_DIR = "signatures"

def load_dataset():
    data = []
    labels = []

    for file in os.listdir(DATA_DIR):
        if file.endswith(".npy"):
            label = file.split("_")[0]
            features = np.load(os.path.join(DATA_DIR, file))

            data.append(features)
            labels.append(label)

    return np.array(data), np.array(labels)

def predict_knn(sample, data, labels, k=3):
    distances = np.linalg.norm(data - sample, axis=1)
    idx = np.argsort(distances)[:k]

    votes = defaultdict(int)
    for i in idx:
        votes[labels[i]] += 1

    return max(votes, key=votes.get)

def main():
    data, labels = load_dataset()

    print("Dataset loaded:", len(data), "samples")

    while True:
        file = input("\nEnter path to sample (.npy): ").strip()
        sample = np.load(file)

        pred = predict_knn(sample, data, labels)
        print("Predicted user:", pred)

if __name__ == "__main__":
    main()
