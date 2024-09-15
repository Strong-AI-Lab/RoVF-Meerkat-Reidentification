import torch
import torch.nn as nn
from transformers import AutoModel

import sys
sys.path.append("..")

from models.perceiver_wrapper import CrossAttention, TransformerEncoder, TransformerDecoder, Perceiver


class RecurrentWrapper(nn.Module):
    def __init__(
        self, perceiver_config: dict, model_name: str, dropout_rate: float = 0.0,
        freeze_image_model: bool=True
    ):
        super(RecurrentWrapper, self).__init__()
        # Load the DINOv2 model
        self.image_model = AutoModel.from_pretrained("facebook/dinov2-small")#model_name)
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
        #print(f"emb_list[0].size(): {emb_list[0].size()}")
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

    def forward(self, video):
        batch_size = video.size(0)

        frame_list = []
        for i in range(video.size(1)):
            frame = video[:,i,:,:,:]    
            frame_output = self.image_model(frame).last_hidden_state
            if self.freeze_image_model:
                frame_output = frame_output.detach()
            frame_output = self.dropout2(frame_output)
            frame_list.append(frame_output)

        #avg_image_emb = self.get_average(frame_list)
        stacked_tensors = torch.stack(frame_list, dim=1) # (b, #frames, sl, dm)
        avg_image_emb = torch.mean(stacked_tensors, dim=-3)
        # (b, sl, dm)
        
        prediction_list = []
        for i in range(video.size(1)):
            frame = video[:,i,:,:,:]
            pred = self.recurrence_model(
                raw_input=frame.permute(0, 2, 3, 1) if self.recurrence_model.use_raw_input else None, 
                #embeddings=frame_list[i], video_emb=avg_image_emb if self.recurrence_model.use_video_emb else None,# if i == 0 else None,
                embeddings=frame_list[i] if self.recurrence_model.use_embeddings else None, 
                video_emb=avg_image_emb if i == 0 else None,
                is_reset_latents=False
            )
            prediction_list.append(pred)

        self.reset_latents()

        return prediction_list
        

def test_perceiver_wrapper():
    
    #raw_input_dim, embedding_dim, latent_dim, num_heads, num_latents, 
        #num_transformer_layers, dropout, output_dim, use_raw_input=True, use_embeddings=True,
        #flatten_channels=False
    perceiver_config = {
        "raw_input_dim": 3,
        "embedding_dim": 384,
        "latent_dim": 384,
        "num_heads": 8,
        "num_latents": 64,
        "num_transformer_layers": 2,
        "dropout": 0.1,
        "output_dim": 768,
        "use_raw_input": True,
        "use_embeddings": True,
        "flatten_channels": False
    }
    dino_model_name = "facebook/dinov2-small"
    dropout_rate = 0.1
    freeze_image_model = True

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
    #print(f"(avg) image_model_test[:20]: {image_model_test[:20]}")
    print()

    image_model_test = model.image_model_forward_and_max(video)
    print(f"(max) image_model_test.size(): {image_model_test.size()}")
    #print(f"(max) image_model_test[:20]: {image_model_test[:20]}")
    print()

    image_model_test = model.image_model_forward_and_concat(video)
    print(f"(cat) image_model_test.size(): {image_model_test.size()}")
    #print(f"(cat) image_model_test[:20]: {image_model_test[:20]}")

if __name__ == "__main__":
    
    test_perceiver_wrapper()