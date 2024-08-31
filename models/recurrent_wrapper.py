import torch
import torch.nn as nn
from transformers import AutoModel

import sys
sys.path.append("..")

from models.perceiver_wrapper import CrossAttention, TransformerEncoder
from models.perceiver_wrapper import Perceiver

class RecurrentWrapper(nn.Module):
    def __init__(
        self, perceiver_config: dict, model_name: str, dropout_rate: float = 0.0,
        freeze_image_model: bool=True
    ):
        super(RecurrentWrapper, self).__init__()
        # Load the DINOv2 model
        self.image_model = AutoModel.from_pretrained(model_name)
        self.recurrence_model = Perceiver(**perceiver_config)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.dropout2 = nn.Dropout(dropout_rate)

        self.freeze_image_model = freeze_image_model

    def reset_latents(self):
        self.recurrence_model.reset_latents()

    def image_model_forward(self, video):
        # video dimensions: batch, #frames, #channels, height, width

        prediction_list = []
        # Process each frame sequentially
        for i in range(video.size(1)):
            
            frame = video[:,i,:,:,:]

            frame_output = self.image_model(frame).last_hidden_state  # Process each frame through DINOv2
            if self.freeze_image_model:
                frame_output = frame_output.detach()
            # (batch_size, slen, d_model)
            frame_output = self.dropout2(frame_output) # TODO: check what dimensions this applies to; it shouldn't matter, but be sure. 
            prediction_list.append(frame_output)

        return prediction_list

    def get_average(self, emb_list):
        '''
        '''
        # emb_list [(batch, slen, d_model), ...]  #length of the list is the number of frames.
        # slen is the number of patches.
        
        stacked_tensors = torch.stack(emb_list, dim=1) # (b, #frames, sl, dm)
        stacked_tensors = self.dropout1(stacked_tensors)
        output_tensor = torch.mean(stacked_tensors, dim=-2) # average along sequence length dimension (each patch)
        # (b, #frames, dm)
        output_tensor = torch.mean(output_tensor, dim=-2)# average along frame dimension
        # (b, dm)
        return output_tensor # (batch_size, d_m)

    def get_max(self, emb_list):
        '''
        '''
        # emb_list [(batch, slen, d_model), ...]  #length of the list is the number of frames.
        # slen is the number of patches.

        stacked_tensors = torch.stack(emb_list, dim=1) # (b, #frames, sl, dm)
        stacked_tensors = self.dropout1(stacked_tensors)
        output_tensor = torch.max(stacked_tensors, dim=-2).values # max along sequence length dimension (each patch)
        # (b, #frames, dm)
        output_tensor = torch.max(output_tensor, dim=-2).values # max along frame dimension
        # (b, dm) 
        return output_tensor

    def concat_embeddings(self, emb_list):
        '''
        '''
        # emb_list [(batch, slen, d_model), ...]  #length of the list is the number of frames.

        output_tensor = torch.cat(emb_list, dim=1) # (b, sl * #frames, dm)
        output_tensor = self.dropout1(output_tensor)
        # flatten.
        #output_tensor = output_tensor.view(-1, emb_list[0].size(1) * len(emb_list) * emb_list[0].size(-1)) #sl*#frames*dm
        output_tensor = output_tensor.view(emb_list[0].size(0), -1)
        return output_tensor # b, sl*#frames*dm

    def image_model_forward_and_average(self, video):
        pre_list = self.image_model_forward(video)
        return self.get_average(pre_list)
    
    def image_model_forward_and_max(self, video):
        pre_list = self.image_model_forward(video)
        return self.get_max(pre_list)
    
    def image_model_forward_and_concat(self, video):
        pre_list = self.image_model_forward(video)
        return self.concat_embeddings(pre_list)

    def recurrence_model_only(self, video_embeddings):
        # Use the final hidden state to make a prediction
        # The video_embeddings is a list of tensors, each tensor is the output of the image model
        # Assuming the output layer is designed to make a prediction based on the hidden state

        # The input to the model is batch_size, seq_len, input_dim

        # video embeddings size [bsize, #frames, 122, 768]

        # The output of the model is batch_size, seq_len, output_dim

        prediction_list = []
        for i in range(video_embeddings.size(1)):
            pred = self.recurrence_model(video_embeddings[:,i,:,:], is_reset_latents=False)
            prediction_list.append(pred)

        self.reset_latents()
        return prediction_list


    def forward(self, video):
        
        batch_size = video.size(0)

        # Process each frame sequentially

        prediction_list = []

        for i in range(video.size(1)):
            
            frame = video[:,i,:,:,:]
            frame_output = self.image_model(frame).last_hidden_state#[:,:,:]  # Process each frame through DINOv2
            if self.freeze_image_model:
                frame_output = frame_output.detach()
            # (batch_size, slen, d_model)
            frame_output = self.dropout2(frame_output)  # Apply dropout and add sequence dimension
            # Update hidden state (and cell state for LSTM)
            #print(f"frame_output.size(): {frame_output.size()}")
            pred = self.recurrence_model(frame_output, is_reset_latents=False)
            prediction_list.append(pred)

        self.reset_latents()

        return prediction_list

def test_perceiver_wrapper():
    
    perceiver_config = {
        "input_dim": 768,
        "latent_dim": 64,
        "num_heads": 8,
        "num_latents": 64,
        "num_transformer_layers": 4,
        "dropout": 0.1,
        "output_dim": 24
    }
    dino_model_name = "facebook/dinov2-base"
    dropout_rate = 0.1
    freeze_image_model = True

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    model = RecurrentWrapper(perceiver_config, dino_model_name, dropout_rate, freeze_image_model).to(device)
    video = torch.randn(2, 8, 3, 224, 224).to(device)
    output = model(video)
    print(f"(rec) len(output): {len(output)}")
    print(f"(rec) output[-1].size(): {output[-1].size()}")
    print()

    ## Gradient test
    loss_fn = nn.MSELoss()
    target = torch.randn_like(output[-1])
    loss = loss_fn(output[-1], target)
    model.zero_grad()
    loss.backward()
    gradients = []
    for param in model.parameters():
        if param.grad is not None:
            gradients.append(param.grad.view(-1))
    gradients = torch.cat(gradients)
    print(f"gradients.size(): {gradients.size()}")
    print(f"Gradients: {gradients}")
    # print out self.latents_p.grad
    print(f"model.recurrence_model.latents_p.grad: {model.recurrence_model.latents_p.grad}")
    print()

    image_model_test = model.image_model_forward_and_average(video)
    print(f"(avg) image_model_test.size(): {image_model_test.size()}")
    print(f"(avg) image_model_test[:20]: {image_model_test[:20]}")
    print()

    image_model_test = model.image_model_forward_and_max(video)
    print(f"(max) image_model_test.size(): {image_model_test.size()}")
    print(f"(max) image_model_test[:20]: {image_model_test[:20]}")
    print()

    image_model_test = model.image_model_forward_and_concat(video)
    print(f"(cat) image_model_test.size(): {image_model_test.size()}")
    print(f"(cat) image_model_test[:20]: {image_model_test[:20]}")

if __name__ == "__main__":
    
    test_perceiver_wrapper()