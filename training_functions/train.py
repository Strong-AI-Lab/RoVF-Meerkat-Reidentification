import torch
import torch.nn as nn

import time
from training_functions.val import val

import transformers
from transformers import AutoModel

import random

def train(
    model, anchor_model, epochs, trainloader, valloader, anchor_fn, similarity_measure, optimizer, scheduler,
    criterion, device, log_path, metadata: str, batch_size=32, clip_value=1.0, start_epoch=0, accumulation_steps=1, 
    margin=1.0
):

    loss_log_path_epoch = f"{log_path}train_epoch_losses.txt" 
    loss_log_path_batch = f"{log_path}train_batch_losses.txt"
    checkpoint_path = f"{log_path}checkpoint_epoch_"

    assert similarity_measure is not None, "similarity_measure must be provided."
    assert batch_size >= 3, "Batch size must be at least 3 to accommodate anchor, positive, and negative examples."
    assert accumulation_steps >= 1, "Accumulation steps must be at least 1."

    model.to(device)
    epoch_loss_list = []
    total_steps = 0

    #margin = 0.0

    for epoch in range(start_epoch, epochs):

        #margin += 0.1
        #print(f"Updated margin to {margin}")
        #criterion.update(margin)

        epoch_start_time = time.time()

        cumulative_loss = 0.0
        batch_counter = 0
        effective_step = 0
        step_loss = 0.0
        batched_input = None
        optimizer.zero_grad()  # Move this outside of the loop to accumulate gradients over steps


        for c, (positive_list, negative_list) in enumerate(trainloader):
            with torch.no_grad(): 
                anchor, positive, negative = anchor_fn(
                    anchor_model if anchor_model is not None else model, positive_list, negative_list, device, similarity_measure=similarity_measure, is_ret_emb=False,
                    margin=margin
                )
                
                if anchor is None:
                    # means no semi-hard examples.
                    continue
            

            '''
            # Randomly select anchor and positive from positive list
            assert len(positive_list) > 1, "Positive list must have at least 2 examples."
            anchor_idx = random.randint(0, len(positive_list) - 1)
            positive_idx = random.randint(0, len(positive_list) - 1)
            while positive_idx == anchor_idx:
                positive_idx = random.randint(0, len(positive_list) - 1)
            anchor = positive_list[anchor_idx].squeeze(0)
            positive = positive_list[positive_idx].squeeze(0)

            # Randomly select a negative from negative list
            negative_idx = random.randint(0, len(negative_list) - 1)
            negative = negative_list[negative_idx].squeeze(0)

            '''

            anchor, positive, negative = anchor.to(device), positive.to(device), negative.to(device)
            #print(f"Anchor: {anchor.size()}, Positive: {positive.size()}, Negative: {negative.size()}")
            if torch.isnan(anchor).any() or torch.isnan(positive).any() or torch.isnan(negative).any():
                print("NaN found in input data")
                continue

            model.train()

            new_data = torch.stack([anchor, positive, negative], dim=0)  # b|3 #frames, #channels, h, w
            batched_input = torch.cat([batched_input, new_data], dim=0) if batched_input is not None else new_data
            batch_counter += 3

            
            if batch_counter > batch_size - 3:

                batched_input = batched_input.to(device)

                emb = model(batched_input)#[-1]
                if isinstance(emb, tuple) or isinstance(emb, list):
                    emb = emb[-1]

                #if len(emb.size()) == 1:
                #    emb = emb.unsqueeze(dim=0)

                # Assuming emb is a 2D tensor with the pattern [anchor, positive, negative, anchor, positive, negative, ...]


                emb_anchor, emb_positive, emb_negative = None, None, None
                if isinstance(emb, tuple) or isinstance(emb, list):
                    for i in range(len(emb)):
                        if len(emb[i].size()) == 1:
                            emb[i] = emb[i].unsqueeze(0)

                        if emb_anchor is None:
                            emb_anchor = emb[i][0::3, :]
                            emb_positive = emb[i][1::3, :]
                            emb_negative = emb[i][2::3, :]
                        else:
                            emb_anchor = torch.cat([emb_anchor, emb[i][0::3, :]], dim=0)
                            emb_positive = torch.cat([emb_positive, emb[i][1::3, :]], dim=0)
                            emb_negative = torch.cat([emb_negative, emb[i][2::3, :]], dim=0)
                else:
                    if len(emb.size()) == 1:
                        emb = emb.unsqueeze(dim=0)
                    emb_anchor = emb[0::3, :]
                    emb_positive = emb[1::3, :]
                    emb_negative = emb[2::3, :]

                if len(emb_anchor.size()) == 1:
                    emb_anchor = emb_anchor.unsqueeze(dim=0)
                    emb_positive = emb_positive.unsqueeze(dim=0)
                    emb_negative = emb_negative.unsqueeze(dim=0)

                loss = criterion(emb_anchor, emb_positive, emb_negative)
                

                if torch.isnan(loss):
                    print(f"NaN found in loss for example {c+1}. Skipping example.")
                    continue

                step_loss = loss.item()
                loss.backward()
            
                if (effective_step + 1) % accumulation_steps == 0:
                    # Gradient clipping (optional)
                    if clip_value is not None:
                        torch.nn.utils.clip_grad_value_(model.parameters(), clip_value=clip_value)
                    optimizer.step()
                    optimizer.zero_grad()  # Zero gradients after every optimizer step
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
            
            emb = model(batched_input)#[-1]
            if isinstance(emb, tuple) or isinstance(emb, list):
                emb = emb[-1]

            emb_anchor, emb_positive, emb_negative = None, None, None
            if isinstance(emb, tuple) or isinstance(emb, list):
                for i in range(len(emb)):
                    if len(emb[i].size()) == 1:
                        emb[i] = emb[i].unsqueeze(0)

                    if emb_anchor is None:
                        emb_anchor = emb[i][0::3, :]
                        emb_positive = emb[i][1::3, :]
                        emb_negative = emb[i][2::3, :]
                    else:
                        emb_anchor = torch.cat([emb_anchor, emb[i][0::3, :]], dim=0)
                        emb_positive = torch.cat([emb_positive, emb[i][1::3, :]], dim=0)
                        emb_negative = torch.cat([emb_negative, emb[i][2::3, :]], dim=0)
            else:
                #if len(emb.size()) == 1:
                #    emb = emb.unsqueeze(dim=0)
                emb_anchor = emb[0::3, :]
                emb_positive = emb[1::3, :]
                emb_negative = emb[2::3, :]

            if len(emb_anchor.size()) == 1:
                emb_anchor = emb_anchor.unsqueeze(dim=0)
                emb_positive = emb_positive.unsqueeze(dim=0)
                emb_negative = emb_negative.unsqueeze(dim=0)

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
            optimizer.zero_grad()
            if scheduler is not None:
                scheduler.step()

            cumulative_loss += step_loss
            effective_step += 1

            print(f"Epoch [{epoch+1}/{epochs}], Step Loss: {step_loss}, Progress: {c+1}/{len(trainloader)} ({((c+1)/len(trainloader))*100:.2f}%)")
            with open(loss_log_path_batch, "a") as loss_log_file:
                loss_log_file.write(f"Epoch [{epoch+1}/{epochs}], Progress: {c+1}/{len(trainloader)}, Step Loss: {step_loss}\n")


        epoch_end_time = time.time()

        avg_loss = cumulative_loss / effective_step if effective_step > 0 else 0
        epoch_loss_list.append(avg_loss)
        print(f"Epoch [{epoch+1}/{epochs}], Average Loss: {avg_loss}, Time: {epoch_end_time - epoch_start_time}")
        with open(loss_log_path_epoch, "a") as loss_log_file:
            loss_log_file.write(f"Epoch [{epoch+1}/{epochs}], Average Loss: {avg_loss}, Time: {epoch_end_time - epoch_start_time}\n")

        # Validation if applicable
        val_start_time, val_end_time = None, None
        val_loss, val_acc_top1, val_acc_top3 = None, None, None
        if valloader is not None:
            val_start_time = time.time()
            val_loss, val_acc_top1, val_acc_top3 = val(
                model, valloader, anchor_fn, device, similarity_measure, criterion, log_path, 
                batch_size, epoch+1, num_epochs=epochs, do_metrics=False, num_negatives=5
            )

            val_end_time = time.time()
            print(f"Validation Loss: {val_loss}, Top-1 Accuracy: {val_acc_top1}, Top-3 Accuracy: {val_acc_top3}, Time: {val_end_time - val_start_time}")

        total_steps += effective_step

        checkpoint = {
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
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
