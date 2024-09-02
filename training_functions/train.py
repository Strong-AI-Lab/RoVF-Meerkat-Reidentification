import torch
import torch.nn as nn

import time
from training_functions.val import val

def train(
    model, model_type, epochs, trainloader, valloader, anchor_fn, similarity_measure, optimizer, scheduler,
    criterion, device, log_path, metadata: str, batch_size=32, clip_value=1.0, start_epoch=0
):

    loss_log_path_epoch = f"{log_path}train_epoch_losses.txt"
    loss_log_path_batch = f"{log_path}train_batch_losses.txt"
    checkpoint_path = f"{log_path}checkpoint_epoch_"

    assert similarity_measure is not None, "similarity_measure must be provided."
    assert batch_size >= 3, "Batch size must be at least 3 to accommodate anchor, positive, and negative examples."

    model.to(device)
    epoch_loss_list = []
    total_steps = 0

    for epoch in range(start_epoch, epochs):

        epoch_start_time = time.time()

        cumulative_loss = 0.0
        batch_counter = 0
        effective_step = 0
        step_loss = 0.0
        batched_input = None

        for c, (positive_list, negative_list) in enumerate(trainloader):
            try:
                with torch.no_grad():
                    anchor, positive, negative = anchor_fn(
                        model, positive_list, negative_list, device, similarity_measure=similarity_measure, is_ret_emb=False
                    )
            except Exception as e:
                print(f"Error in anchor_fn for example {c+1}. Skipping example.")
                print(e)
                continue
            anchor, positive, negative = anchor.to(device), positive.to(device), negative.to(device)

            model.train()

            new_data = torch.stack([anchor, positive, negative], dim=0) # b|3 #frames, #channels, h, w
            batched_input = torch.cat([batched_input, new_data], dim=0) if batched_input is not None else new_data
            batch_counter += 3

            if batch_counter > batch_size - 3:

                batched_input = batched_input.to(device)
                #print(f"batched_input.size(): {batched_input.size()}")
                optimizer.zero_grad()
                emb = None
                if "recurrent" in model_type:
                    #emb = model(batched_input.permute(0, 1, 4, 2, 3))[-1]
                    emb = model(batched_input)[-1]
                else:
                    #emb = model(batched_input.permute(0, 1, 4, 2, 3))
                    emb = model(batched_input)
                # emb.size() = (batch_size, emb_dim)
                #print(f"emb.size(): {emb.size()}")

                # Assuming emb is a 2D tensor with the pattern [anchor, positive, negative, anchor, positive, negative, ...]
                emb_anchor = emb[0::3, :]
                emb_positive = emb[1::3, :]
                emb_negative = emb[2::3, :]

                if len(emb_anchor.size()) == 1:
                    emb_anchor = emb_anchor.unsqueeze(dim=0)
                    emb_positive = emb_positive.unsqueeze(dim=0)
                    emb_negative = emb_negative.unsqueeze(dim=0)

                loss = criterion(emb_anchor, emb_positive, emb_negative)
                step_loss = loss.item()
                loss.backward()
                if clip_value is not None:
                    torch.nn.utils.clip_grad_value_(model.parameters(), clip_value=clip_value)
                optimizer.step()
                if scheduler is not None:
                    scheduler.step()

                cumulative_loss += step_loss
                effective_step += 1
                batched_input = None
                batch_counter = 0
                
                print(f"Epoch [{epoch+1}/{epochs}], Step Loss: {step_loss}, Progress: {c+1}/{len(trainloader)} ({((c+1)/len(trainloader))*100:.2f}%)")
                with open(loss_log_path_batch, "a") as loss_log_file:
                    loss_log_file.write(f"Epoch [{epoch+1}/{epochs}], Progress: {c+1}/{len(trainloader)}, Step Loss: {step_loss}\n")

        if batched_input is not None:  # Process the final, possibly incomplete batch
            
            batched_input = batched_input.to(device)
            optimizer.zero_grad()
            if "recurrent" in model_type:
                #emb = model(batched_input.permute(0, 1, 4, 2, 3))[-1]
                emb = model(batched_input)[-1]
            else: # dino model
                emb = model(batched_input.permute(0, 1, 4, 2, 3))
                emb = model(batched_input)

            # Assuming emb is a 2D tensor with the pattern [anchor, positive, negative, anchor, positive, negative, ...]
            emb_anchor = emb[0::3, :]
            emb_positive = emb[1::3, :]
            emb_negative = emb[2::3, :]

            if len(emb_anchor.size()) == 1:
                emb_anchor = emb_anchor.unsqueeze(dim=0)
                emb_positive = emb_positive.unsqueeze(dim=0)
                emb_negative = emb_negative.unsqueeze(dim=0)

            loss = criterion(emb_anchor, emb_positive, emb_negative)
            loss.backward()
            if clip_value is not None:
                torch.nn.utils.clip_grad_value_(model.parameters(), clip_value=clip_value)
            optimizer.step()
            if scheduler is not None:
                scheduler.step()

            cumulative_loss += step_loss
            effective_step += 1

            # Note: len(trainloader) is correct as this is the epoch complete.
            print(f"Epoch [{epoch+1}/{epochs}], Step Loss: {step_loss}, Progress: {len(trainloader)}/{len(trainloader)} ({(len(trainloader)/len(trainloader))*100:.2f}%)")
            with open(loss_log_path_batch, "a") as loss_log_file:
                loss_log_file.write(f"Epoch [{epoch+1}/{epochs}], Effective Step: {len(trainloader)}/{len(trainloader)}, Step Loss: {step_loss}\n")
            # TODO: Effective Step / len(trainloader) is not accurate. train loader should be divided by items in a batch... 


        epoch_end_time = time.time()

        #if effective_step != len(trainloader):
        #    print(f"WARNING: effective_step ({effective_step}) != len(trainloader) ({len(trainloader)})")

        avg_loss = cumulative_loss / effective_step if effective_step > 0 else 0
        epoch_loss_list.append(avg_loss)
        print(f"Epoch [{epoch+1}/{epochs}], Average Loss: {avg_loss}, Time: {epoch_end_time - epoch_start_time}")
        with open(loss_log_path_epoch, "a") as loss_log_file:
            loss_log_file.write(f"Epoch [{epoch+1}/{epochs}], Average Loss: {avg_loss}, Time: {epoch_end_time - epoch_start_time}\n")
        
        val_start_time, val_end_time = None, None
        val_loss, val_acc_top1, val_acc_top3 = None, None, None
        if valloader is not None:
            val_start_time = time.time()
            val_loss, val_acc_top1, val_acc_top3 = val(
                model, model_type, valloader, device, anchor_fn, similarity_measure, criterion, log_path, batch_size, epoch+1, num_epochs=epochs
            )

            val_end_time = time.time()
            print(f"Validation Loss: {val_loss}, Top-1 Accuracy: {val_acc_top1}, Top-3 Accuracy: {val_acc_top3}, Time: {val_end_time - val_start_time}")

        total_steps += effective_step

        checkpoint = {
            'model_state_dict': model.state_dict(),
            'metadata': metadata, 
            'epoch': epoch+1,
            'number_individual_examples': total_steps,
            'train_epoch_time': epoch_start_time - epoch_end_time,
            'epoch_train_loss': avg_loss,
            'epoch_val_loss': val_loss,
            'epoch_val_acc_top1': val_acc_top1,
            'epoch_val_acc_top3': val_acc_top3,
            'epoch_val_time': val_start_time - val_end_time if val_start_time is not None else None
        }
        tmp_ckpt_path = checkpoint_path+str(epoch+1)+".pt"
        torch.save(checkpoint, tmp_ckpt_path)
        print(f"Checkpoint saved to {tmp_ckpt_path}")


