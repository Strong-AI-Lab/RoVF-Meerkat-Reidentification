import pickle
import numpy as np
import pandas as pd

def open_pickle(filepath):
    """Open and load a pickle file."""
    with open(filepath, 'rb') as f:
        return pickle.load(f)

def indices_of_smallest(distances, banned_idx):
    #Sort indices and distances, ignoring the query
    sorted_indices = np.argsort(distances)
    sorted_indices = sorted_indices[sorted_indices != banned_idx]
    sorted_distances = distances[sorted_indices]
    
    # Find the lowest 3 distinct values
    unique_values = []
    unique_indices = []
    
    for i in range(len(sorted_distances) - 1):
        if sorted_distances[i] != sorted_distances[i + 1]:
            unique_values.append(sorted_distances[i])
            unique_indices.append(sorted_indices[i])
        if len(unique_values) == 3:
            break
    
    # Check if the next value after the last unique one is the same (indicating a tie)
    if len(unique_values) == 3 and i+1 < len(sorted_distances):
        if sorted_distances[i+1] == sorted_distances[i]:
            # There's a tie for the last included value, so we remove any tied values
            last_val = unique_values[-1]
            return [idx for idx, val in zip(unique_indices, unique_values) if val != last_val]
    
    return unique_indices

def compute_distances(embeddings):
    """Compute Euclidean distances between embeddings."""
    return np.sqrt(((embeddings[:, np.newaxis, :] - embeddings[np.newaxis, :, :]) ** 2).sum(axis=2))

def get_metrics(models, df):
    """
    Compute top-1, top-3 accuracy and the number of unique elements for each model.

    Parameters:
    - models: List of embedding file paths.
    - df: DataFrame with test examples.

    Returns:
    - A list of metric results for each model: top-1 accuracy, top-3 accuracy, and unique top-3 counts.
    """
    
    # Load all model embeddings
    data = []
    for m in models:
        dict_ = open_pickle(m)
        for key in dict_:
            dict_[key] = dict_[key].to("cpu")
        data.append(dict_)

    # Initialize results array: [models x rows x metrics]
    results = np.zeros((len(models), len(df)*2, 3))
    
    idx = 0

    # Loop through each row in the dataframe
    for index, row in df.iterrows():
        for k, m in enumerate(data):
            row_embeddings = []
            for i in row:
                row_embeddings.append(m[i].cpu().numpy())
            embeddings = np.stack(row_embeddings)

            # Compute Euclidean distances between embeddings
            distances = compute_distances(embeddings)

            # Get indices of closest embeddings (excluding the query)
            closest_indices_ranked_0 = indices_of_smallest(distances[0], 0)
            closest_indices_ranked_1 = indices_of_smallest(distances[1], 1)

            for inc, ranks in enumerate([closest_indices_ranked_0, closest_indices_ranked_1]):
                if ranks:
                    # Top-1 accuracy
                    if ranks[0] in [0, 1]:
                        results[k, idx + inc, 0] = 1
                    # Top-3 accuracy
                    if 0 in ranks[:3] or 1 in ranks[:3]:
                        results[k, idx + inc, 1] = 1
                # Number of unique elements in top-3
                results[k, idx + inc, 2] = len(ranks)

        idx += 2

    # Collect and return average metrics for each model
    metrics = []
    for m in range(len(data)):
        top1 = np.mean(results[m, :, 0])  # Average top-1 accuracy
        top3 = np.mean(results[m, :, 1])  # Average top-3 accuracy
        unique_top3 = np.mean(results[m, :, 2])  # Average number of unique elements in top-3
        metrics.append((top1, top3, unique_top3))

    return metrics

def main():
    # List of model embedding paths
    models = [
        "/home/kkno604/github/meerkat-repos/RoVF-meerkat-reidentification/results/hyperparameter_search/rovf_margin_0p5/checkpoint_epoch_5_embeddings.pkl"
    ]
    
    # Load dataframe of test examples
    df = pd.read_csv("/home/kkno604/github/meerkat-repos/RoVF-meerkat-reidentification/Dataset/meerkat_h5files/Precomputed_test_examples_meerkat.csv")
    
    # Get metrics for all models
    metrics = get_metrics(models, df)
    
    # Print the results
    for i, (top1, top3, unique_top3) in enumerate(metrics):
        print(f"Model {i}: Top-1 Accuracy: {top1}, Top-3 Accuracy: {top3}, Unique in Top-3: {unique_top3}")

if __name__ == "__main__":
    main()
