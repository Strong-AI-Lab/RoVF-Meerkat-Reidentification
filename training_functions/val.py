import torch
import torch.nn as nn

def val(
    model, model_type, valloader, device, anchor_fn, similarity_measure, 
    criterion, log_path, batch_size, current_epoch, num_epochs, do_metrics=False
):

    assert similarity_measure is not None, "similarity_measure must be provided."

    def get_top1_and_top3(distances):
        assert len(distances.size()) == 1 or len(distances.squeeze().size()) == 1, "distances must be a 1D tensor (or equivalent)."
        top_1 = torch.argmin(distances).item() # Returns the indices of the minimum value(s) of the flattened tensor or along a dimension
        top_3 = torch.topk(distances, 3, largest=False).indices
        
        top_1_correct = 1 if top_1 == 0 else 0
        top_3_correct = 1 if 0 in top_3 else 0
        
        top_1_total = 1
        top_3_total = 1
        return top_1_correct, top_1_total, top_3_correct, top_3_total

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
        
        try:
            with torch.no_grad():
                anchor, positive, negative, negative_list = anchor_fn(
                    model, positive_list, negative_list, device, similarity_measure=similarity_measure, is_ret_emb=True
                )
        except Exception as e:
            print(f"Error in anchor_fn for example {c+1}. Skipping example.")
            print(e)
            continue
        anchor, positive, negative = anchor.to(device), positive.to(device), negative.to(device)
        negative_list = [neg.to(device) for neg in negative_list]


        loss = criterion(anchor, positive, negative)
        cumulative_loss += loss.item()
        counter += 1

        if not do_metrics:
            continue

        # get top-1 and top-3 accuracy.
        query = anchor # (d_model)
        gallery = torch.stack([positive]+negative_list, dim=0)
        distances = similarity_measure(query.unsqueeze(0), gallery) # .size() = (1 + len(negative_list))
        assert len(distances.size()) == 1, "distances must be a 1D tensor."
        a, b, c, d, = get_top1_and_top3(distances)
        top_1_correct += a
        top_1_total += b
        top_3_correct += c
        top_3_total += d

        # switch the query to the positive (non-anchor) example.
        query = positive # (d_model)
        gallery = torch.stack([anchor]+negative_list, dim=0) # b, d_model
        distances = similarity_measure(query.unsqueeze(0), gallery) # .size() = (1 + len(negative_list))
        assert len(distances.size()) == 1, "distances must be a 1D tensor."
        a, b, c, d, = get_top1_and_top3(distances)
        top_1_correct += a
        top_1_total += b
        top_3_correct += c
        top_3_total += d

    
    avg_loss = cumulative_loss / counter
    if not do_metrics:
        print(f"Epoch [{current_epoch}/{num_epochs}], Average Val Loss: {avg_loss}")
        with open(loss_log_path_epoch, "a") as loss_log_file:
            loss_log_file.write(f"Epoch [{current_epoch}/{num_epochs}], Average Loss: {avg_loss}\n")
        return avg_loss, None, None
    print(f"Epoch [{current_epoch}/{num_epochs}], Average Val Loss: {avg_loss}, Top-1 Total Correct: {top_1_correct}/{top_1_total} ({top_1_correct/top_1_total}), Top-3 Total Correct: {top_3_correct}/{top_3_total} ({top_3_correct/top_3_total})")
    with open(loss_log_path_epoch, "a") as loss_log_file:
        loss_log_file.write(f"Epoch [{current_epoch}/{num_epochs}], Average Loss: {avg_loss}, Top-1 Total Correct: {top_1_correct}/{top_1_total} ({top_1_correct/top_1_total}), Top-3 Total Correct: {top_3_correct}/{top_3_total} ({top_3_correct/top_3_total})\n")

    return avg_loss, top_1_correct/top_1_total, top_3_correct/top_3_total