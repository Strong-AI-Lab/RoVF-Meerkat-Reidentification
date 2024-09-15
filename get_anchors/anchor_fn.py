import torch
import torch.nn as nn
import random

def anchor_fn_hard_rand_anchor(model, positive_list, negative_list, device, similarity_measure=None, is_ret_emb=False, margin=1.0):
    assert len(positive_list) >= 2, "Positive list must contain at least 2 elements."

    model.eval()

    if similarity_measure is None:
        similarity_measure = lambda anchor, other: torch.sqrt(torch.sum((anchor - other) ** 2, dim=-1))

    # Concatenate all positive and negative examples into a single tensor and move to device.
    batch_tens = torch.cat(positive_list + negative_list, dim=0).to(device)

    # Get embeddings for all samples
    emb_tens = model(batch_tens)
    if isinstance(emb_tens, tuple) or isinstance(emb_tens, list):
        emb_tens = emb_tens[-1]

    # Randomly choose an anchor from positive list
    anchor_idx = random.randint(0, len(positive_list) - 1)
    anchor_emb = emb_tens[anchor_idx]

    # Calculate distances from anchor to all other positives
    positive_embeds = emb_tens[:len(positive_list)]
    dist_to_anchor = similarity_measure(positive_embeds, anchor_emb.unsqueeze(0)).squeeze()
    dist_to_anchor[anchor_idx] = float('inf')  # Exclude the anchor itself

    # Find the furthest positive
    positive_idx = torch.argmax(dist_to_anchor)

    # Calculate distances from anchor to all negatives
    negative_embeds = emb_tens[len(positive_list):]
    dist_to_negatives = similarity_measure(negative_embeds, anchor_emb.unsqueeze(0)).squeeze()

    # Find the closest negative
    negative_idx = torch.argmin(dist_to_negatives) + len(positive_list)

    if not is_ret_emb:
        anchor = positive_list[anchor_idx].squeeze()
        positive = positive_list[positive_idx].squeeze()
        negative = negative_list[negative_idx - len(positive_list)].squeeze()
        return anchor, positive, negative
    else:  # is_ret_emb; this is the case for validation/testing
        anchor = emb_tens[anchor_idx]
        positive = emb_tens[positive_idx]
        negative = emb_tens[negative_idx]
        negative_list = [emb_tens[z] for z in range(len(positive_list), len(positive_list) + len(negative_list))]
        return anchor, positive, negative, negative_list


def anchor_fn_hard(model, positive_list, negative_list, device, similarity_measure=None, is_ret_emb=False, margin=1.0):
    
    assert len(positive_list) >= 2, "Positive list is empty."

    #model.train()
    model.eval()

    # Set similarity measure if it's not provided.
    if similarity_measure is None:
        similarity_measure = lambda anchor, other: torch.sqrt(torch.sum((anchor - other) ** 2, dim=-1))

    # Reset latents of the recurrent model once per batch.
    #model.reset_latents()

    # Concatenate all positive and negative examples into a single tensor and move to device.
    batch_tens = torch.cat(positive_list + negative_list, dim=0).to(device)  # Shape: (batch, seq_len, height, width, channels)
    #print(f"batch_tens.size(): {batch_tens.size()}")

    # Get a list of embeddings for each frame in the video from the recurrent model.
    #video_list = model(batch_tens.permute(0, 1, 4, 2, 3))
    emb_tens = model(batch_tens)
    if isinstance(emb_tens, tuple) or isinstance(emb_tens, list):
        emb_tens = emb_tens[-1]
    #emb Shape: (batch, d_model) 

    # Reset latents for the next batch.
    #model.reset_latents()

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

    #dist_between_positives = similarity_measure(positive_embeds[pos1].unsqueeze(0), positive_embeds[pos2].unsqueeze(0)).item()

    # Choose anchor and positive based on smaller distance to their respective closest negatives.
    if min_neg_distance_for_pos1 < min_neg_distance_for_pos2:
        anchor_idx, positive_idx = pos1, pos2
        negative_idx = neg_for_pos1
        #dist_to_negative = min_neg_distance_for_pos1.item()
    else:
        anchor_idx, positive_idx = pos2, pos1
        negative_idx = neg_for_pos2
        #dist_to_negative = min_neg_distance_for_pos2.item()

    # Check if the triplet satisfies the margin condition
    #print(f"dist_between_positives: {dist_between_positives} dist_to_negative: {dist_to_negative}, margin: {margin}")
    #if dist_to_negative >= dist_between_positives + margin:
    #    if not is_ret_emb:
    #        return None, None, None
    #    else:
    #        return None, None, None, None

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




# semi-hard
def anchor_fn_semi_hard(model, positive_list, negative_list, device, margin=0.3, similarity_measure=None, is_ret_emb=False): 

    def select_negative(anchor_embed, positive_embed, negative_embeds, temperature=1.0):
        dist_to_positive = torch.norm(anchor_embed - positive_embed)
        dist_to_negatives = torch.norm(negative_embeds - anchor_embed, dim=1)
        
        # Calculate scores: higher for semi-hard negatives
        scores = torch.exp((dist_to_negatives - dist_to_positive) / temperature)
        
        # Normalize scores to probabilities
        probs = scores / scores.sum()
        
        # Sample negative index based on probabilities
        negative_idx = torch.multinomial(probs, 1).item()
        
        return negative_idx, negative_embeds[negative_idx]

    assert len(positive_list) >= 2, "Positive list is empty."
    model.eval()

    # Set similarity measure if it's not provided.
    if similarity_measure is None:
        similarity_measure = lambda anchor, other: torch.sqrt(torch.sum((anchor - other) ** 2, dim=-1))

    # Concatenate all positive and negative examples into a single tensor and move to device.
    batch_tens = torch.cat(positive_list + negative_list, dim=0).to(device)  # Shape: (batch, seq_len, height, width, channels)

    # Get embeddings for each example using the model.
    emb_tens = model(batch_tens)
    if isinstance(emb_tens, tuple) or isinstance(emb_tens, list):
        emb_tens = emb_tens[-1]  # If model returns a tuple or list, get the last element.

    # Get positive and negative embeddings.
    positive_embeds = emb_tens[:len(positive_list)]  # Shape: (num_positives, d_model)
    negative_embeds = emb_tens[len(positive_list):]  # Shape: (num_negatives, d_model)

    # Calculate pairwise distances between all positive embeddings.
    dist_matrix = similarity_measure(positive_embeds.unsqueeze(1), positive_embeds.unsqueeze(0))

    # Mask the diagonal (distance between an element and itself) to avoid selecting it.
    dist_matrix.fill_diagonal_(float('-inf'))  # .size() = (num_positives, num_positives)

    # Find the two positives with the largest distance.
    max_distance, max_indices = torch.max(dist_matrix, dim=1)
    pos1, pos2 = torch.argmax(max_distance), max_indices[torch.argmax(max_distance)]

    # Select the positives that will serve as anchor and positive.
    #anchor_idx = torch.randint(0, len(positive_list), (1,))
    #positive_idx = torch.randint(0, len(positive_list), (1,))
    #while anchor_idx == positive_idx:
    #    positive_idx = torch.randint(0, len(positive_list), (1,))
    
    
    # Find the two positives with the largest distance.
    #max_distance, max_indices = torch.max(dist_matrix, dim=1)
    #pos1, pos2 = torch.argmax(max_distance), max_indices[torch.argmax(max_distance)]

    # Select the positives that will serve as anchor and positive.
    anchor_idx, positive_idx = pos1, pos2

    # Calculate distances between the selected anchor and all negatives.
    dist_to_negatives = similarity_measure(negative_embeds, positive_embeds[anchor_idx].unsqueeze(0)).squeeze()
    dist_to_positive = similarity_measure(positive_embeds[anchor_idx].unsqueeze(0), positive_embeds[positive_idx].unsqueeze(0)).squeeze()
    
    #print(f"dist_to_negatives.size(): {dist_to_negatives.size()}, dist_to_positive.size(): {dist_to_positive.size()}")
    #assert len(dist_to_negatives.size()) == 1, f"dist_to_negatives must be a 1D tensor {dist_to_negatives.size()}."
    #assert len(dist_to_positive.size()) == 0, f"dist_to_positive must be a 1D tensor. {dist_to_positive.size()}"

    # Semi-Hard Negative Selection with Margin
    # We are looking for negatives where: dist_to_neg > dist_to_pos and dist_to_neg < dist_to_pos + margin (semi-hard).
    semi_hard_negatives_mask = (dist_to_negatives > dist_to_positive) & (dist_to_negatives < dist_to_positive + margin)
    #print(f"semi_hard_negatives_mask.size(): {semi_hard_negatives_mask.size()}")
    #print(f"semi_hard_negatives_mask: {semi_hard_negatives_mask}")
    semi_hard_negatives = dist_to_negatives[semi_hard_negatives_mask]
    #print(f"semi_hard_negatives.size(): {semi_hard_negatives.size()}")

    

    if semi_hard_negatives.numel() > 0:
        # If there are semi-hard negatives, choose the closest semi-hard negative.
        min_neg_distance, neg_for_anchor = torch.min(semi_hard_negatives, dim=0)
        # Get the original index of the selected semi-hard negative.
        neg_for_anchor_idx = torch.nonzero(semi_hard_negatives_mask)[neg_for_anchor].item()
    else:
        # Fallback to using the hardest negative (closest negative) if no semi-hard negatives found.
        min_neg_distance, neg_for_anchor_idx = torch.min(dist_to_negatives, dim=0)
        #if not is_ret_emb:
        #    return None, None, None
        #else:
        #    return None, None, None, None

    # Adjust index to match the original batch index for the negative.
    negative_idx = neg_for_anchor_idx + len(positive_list)

    # Ensure indices for anchor, positive, and negative are distinct.
    assert negative_idx != positive_idx and negative_idx != anchor_idx and positive_idx != anchor_idx, f"Indices are not distinct: {anchor_idx}, {positive_idx}, {negative_idx}, {len(positive_list)}, {len(negative_list)}"

    if not is_ret_emb:
        # Return the original tensor data instead of embeddings.
        anchor = positive_list[anchor_idx].squeeze()
        positive = positive_list[positive_idx].squeeze()
        negative = negative_list[negative_idx - len(positive_list)].squeeze()  # Adjust index for negative_list.
        return anchor, positive, negative
    else:  # Return embeddings; this is the case for validation/testing
        anchor = emb_tens[anchor_idx]
        positive = emb_tens[positive_idx]
        negative = emb_tens[negative_idx]
        # Return the anchor, positive, negative embeddings, and the list of negative embeddings.
        negative_list_emb = [emb_tens[z] for z in range(len(positive_list), len(positive_list) + len(negative_list))]
        return anchor, positive, negative, negative_list_emb
