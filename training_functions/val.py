import torch


def _embedding_matrix(value, name):
    if isinstance(value, torch.Tensor):
        if value.dim() == 1:
            return value.unsqueeze(0)
        if value.dim() == 2:
            return value
        raise ValueError(f"{name} must be a 1D or 2D tensor, got shape {tuple(value.size())}.")

    if isinstance(value, (list, tuple)):
        if not value:
            raise ValueError(f"{name} must contain at least one embedding.")
        matrices = [_embedding_matrix(item, f"{name}[{idx}]") for idx, item in enumerate(value)]
        return torch.cat(matrices, dim=0)

    raise ValueError(f"{name} must be a tensor or a list/tuple of tensors.")


def _require_same_embedding_dim(reference, candidate, reference_name, candidate_name):
    if reference.size(1) != candidate.size(1):
        raise ValueError(
            f"{candidate_name} embedding dim {candidate.size(1)} does not match "
            f"{reference_name} embedding dim {reference.size(1)}."
        )


def _top1_top3_correct(query, gallery, similarity_measure):
    distances = similarity_measure(query, gallery)
    if not isinstance(distances, torch.Tensor):
        raise ValueError("similarity_measure must return a tensor.")
    distances = distances.squeeze()
    if distances.dim() != 1:
        raise ValueError("distances must be a 1D tensor.")

    ranked_indices = torch.argsort(distances)
    top_1_correct = 1 if ranked_indices[0].item() == 0 else 0
    top_3_correct = 1 if 0 in ranked_indices[:3].tolist() else 0
    return top_1_correct, top_3_correct


def _require_min_length(value, name, minimum):
    try:
        actual = len(value)
    except TypeError as exc:
        raise ValueError(f"{name} must have at least {minimum} items.") from exc
    if actual < minimum:
        raise ValueError(f"{name} must have at least {minimum} items.")


def val(
    model, valloader, anchor_fn, device, similarity_measure, criterion, 
    log_path, batch_size, current_epoch, num_epochs, do_metrics=False, 
    num_negatives=5
):
    _ = batch_size, num_negatives
    if similarity_measure is None:
        raise ValueError("similarity_measure must be provided.")

    # current_epoch should have 1 added already.

    loss_log_path_epoch = f"{log_path}val_epoch_losses.txt"

    model.to(device)

    cumulative_loss = 0.0
    counter = 0

    top_1_correct = 0
    top_1_total = 0

    top_3_correct = 0
    top_3_total = 0

    for c, (positive_list, negative_list) in enumerate(valloader):

        _require_min_length(positive_list, "Positive list", 2)
        _require_min_length(negative_list, "Negative list", 1)
        
        with torch.no_grad():
            anchor_emb, positive_emb, selected_negative_emb, all_negative_embs = anchor_fn(
                model, positive_list, negative_list, device, similarity_measure=similarity_measure, is_ret_emb=True
            )

        anchor_emb = _embedding_matrix(anchor_emb, "anchor_emb")
        positive_emb = _embedding_matrix(positive_emb, "positive_emb")
        selected_negative_emb = _embedding_matrix(selected_negative_emb, "selected_negative_emb")
        negative_gallery = (
            selected_negative_emb
            if all_negative_embs is None
            else _embedding_matrix(all_negative_embs, "all_negative_embs")
        )

        _require_same_embedding_dim(anchor_emb, positive_emb, "anchor_emb", "positive_emb")
        _require_same_embedding_dim(anchor_emb, selected_negative_emb, "anchor_emb", "selected_negative_emb")
        _require_same_embedding_dim(anchor_emb, negative_gallery, "anchor_emb", "all_negative_embs")
        if anchor_emb.size() != positive_emb.size() or anchor_emb.size() != selected_negative_emb.size():
            raise ValueError(
                "anchor_emb, positive_emb, and selected_negative_emb must have matching shapes "
                f"for triplet loss; got {tuple(anchor_emb.size())}, {tuple(positive_emb.size())}, "
                f"{tuple(selected_negative_emb.size())}."
            )

        loss = criterion(anchor_emb, positive_emb, selected_negative_emb)
        cumulative_loss += loss.item()
        counter += 1

        if not do_metrics:
            continue

        if negative_gallery.size(0) < 2:
            raise ValueError("Metric mode requires at least 2 negative embeddings for Top-3.")

        for idx in range(anchor_emb.size(0)):
            anchor_gallery = torch.cat([positive_emb[idx].unsqueeze(0), negative_gallery], dim=0)
            positive_gallery = torch.cat([anchor_emb[idx].unsqueeze(0), negative_gallery], dim=0)

            a_top1, a_top3 = _top1_top3_correct(anchor_emb[idx], anchor_gallery, similarity_measure)
            p_top1, p_top3 = _top1_top3_correct(positive_emb[idx], positive_gallery, similarity_measure)
            top_1_correct += a_top1 + p_top1
            top_1_total += 2
            top_3_correct += a_top3 + p_top3
            top_3_total += 2

    if counter == 0:
        raise ValueError("Validation loader produced no batches.")
    avg_loss = cumulative_loss / counter
    if not do_metrics:
        print(f"Epoch [{current_epoch}/{num_epochs}], Average Val Loss: {avg_loss}")
        with open(loss_log_path_epoch, "a") as loss_log_file:
            loss_log_file.write(f"Epoch [{current_epoch}/{num_epochs}], Average Loss: {avg_loss}\n")
        return avg_loss, None, None
    top_1_acc = top_1_correct/top_1_total
    top_3_acc = top_3_correct/top_3_total
    print(f"Epoch [{current_epoch}/{num_epochs}], Average Val Loss: {avg_loss}, Top-1 Total Correct: {top_1_correct}/{top_1_total} ({top_1_acc}), Top-3 Total Correct: {top_3_correct}/{top_3_total} ({top_3_acc})")
    with open(loss_log_path_epoch, "a") as loss_log_file:
        loss_log_file.write(f"Epoch [{current_epoch}/{num_epochs}], Average Loss: {avg_loss}, Top-1 Total Correct: {top_1_correct}/{top_1_total} ({top_1_acc}), Top-3 Total Correct: {top_3_correct}/{top_3_total} ({top_3_acc})\n")

    return avg_loss, top_1_acc, top_3_acc
