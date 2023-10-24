import torch

def mapk(actual, predicted, k=10):
    """
    Calculates Mean Average Precision at k.
    actual: A list of lists, where each sublist contains the true items for a query.
    predicted: A list of lists, where each sublist contains the predicted items for a query.
    k: The maximum number of predicted items to consider.
    """
    # Calculate precision at each position up to k
    aps = []
    for i in range(len(actual)):
        ap = 0.0
        num_hits = 0.0
        for j in range(k):
            if predicted[i][j] in actual[i]:
                num_hits += 1.0
                ap += num_hits / (j + 1.0)
        if num_hits > 0.0:
            ap /= num_hits
        aps.append(ap)

    # Calculate the mean average precision
    mapk = torch.tensor(aps).mean().item()
    return mapk

if __name__ == '__main__':
    # Actual items for each query
    actual = [[1, 2, 3], [0, 3], [2], [1, 2, 3, 4], [4]]

    # Predicted items for each query
    predicted = [[0, 1, 2, 3, 4], [0, 1, 2], [1, 2, 3], [0, 1, 2, 3], [0, 1, 2]]

    # Calculate MAP@3
    mapk_score = mapk(actual, predicted, k=3)

    print(f"MAP@3: {mapk_score:.4f}")