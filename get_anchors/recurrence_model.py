import torch
import torch.nn as nn

def get_anchor_pos_and_neg_recurrence(model, positive_list, negative_list, device, similarity_measure=None, is_ret_emb=False):
    
    assert len(positive_list) >= 2, "Positive list is empty."

    model.eval()

    # Set similarity measure if it's not provided.
    if similarity_measure is None:
        similarity_measure = lambda anchor, other: torch.sqrt(torch.sum((anchor - other) ** 2, dim=-1))

    # Reset latents of the recurrent model once per batch.
    model.reset_latents()

    # Concatenate all positive and negative examples into a single tensor and move to device.
    batch_tens = torch.cat(positive_list + negative_list, dim=0).to(device)  # Shape: (batch, seq_len, height, width, channels)
    #print(f"batch_tens.size(): {batch_tens.size()}")

    # Get a list of embeddings for each frame in the video from the recurrent model.
    #video_list = model(batch_tens.permute(0, 1, 4, 2, 3))
    video_list = model(batch_tens)
    emb_tens = video_list[-1]  # Shape: (batch, d_model)

    # Reset latents for the next batch.
    model.reset_latents()

    # Calculate pairwise distances between all positive embeddings.
    positive_embeds = emb_tens[:len(positive_list)]  # Shape: (num_positives, d_model)
    dist_matrix = similarity_measure(positive_embeds.unsqueeze(1), positive_embeds.unsqueeze(0))

    # Mask the diagonal (distance between an element and itself) to avoid selecting it.
    dist_matrix.fill_diagonal_(float('-inf')) # .size() = (num_positives, num_positives)

    # Find the two positives with the largest distance.
    max_distance, max_indices = torch.max(dist_matrix, dim=1)
    pos1, pos2 = torch.argmax(max_distance), max_indices[torch.argmax(max_distance)]

    # Calculate distances between selected positives and all negatives.
    negative_embeds = emb_tens[len(positive_list):]  # Shape: (num_negatives, d_model)
    dist_to_pos1 = similarity_measure(negative_embeds, positive_embeds[pos1].unsqueeze(0)).squeeze()
    dist_to_pos2 = similarity_measure(negative_embeds, positive_embeds[pos2].unsqueeze(0)).squeeze()

    # Find the closest negative for each positive.
    min_neg_distance_for_pos1, neg_for_pos1 = torch.min(dist_to_pos1, dim=0)
    min_neg_distance_for_pos2, neg_for_pos2 = torch.min(dist_to_pos2, dim=0)

    # Adjust the indices for the negative examples.
    neg_for_pos1 += len(positive_list)
    neg_for_pos2 += len(positive_list)

    # Choose anchor and positive based on smaller distance to their respective closest negatives.
    if min_neg_distance_for_pos1 < min_neg_distance_for_pos2:
        anchor_idx, positive_idx = pos1, pos2
        negative_idx = neg_for_pos1
    else:
        anchor_idx, positive_idx = pos2, pos1
        negative_idx = neg_for_pos2

    assert negative_idx != positive_idx and negative_idx != anchor_idx and positive_idx != anchor_idx, f"Indices are not distinct: {anchor_idx}, {positive_idx}, {negative_idx}, {len(positive_list)}, {len(negative_list)}"

    if not is_ret_emb:
        anchor = positive_list[anchor_idx].squeeze()
        positive = positive_list[positive_idx].squeeze()
        negative = negative_list[negative_idx - len(positive_list)].squeeze()  # Adjust index for negative_list.
        return anchor, positive, negative
    else: # is_ret_emb; this is the case for validation/testing
        anchor = emb_tens[anchor_idx]
        positive = emb_tens[positive_idx]
        negative = emb_tens[negative_idx]
        negative_list = [emb_tens[z] for z in range(len(positive_list), len(positive_list) + len(negative_list))]
        return anchor, positive, negative, negative_list