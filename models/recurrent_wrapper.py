import torch
import torch.nn as nn
from transformers import AutoModel

import sys
sys.path.append("..")

from models.perceiver_wrapper import CrossAttention, TransformerEncoder, TransformerDecoder

class RecurrentWrapper(nn.Module):
    def __init__(
        self, perceiver_config: dict, model_name: str, dropout_rate: float = 0.0,
        freeze_image_model: bool=True, is_append_avg_emb: bool=False, type_="v1", recurrent_type="perceiver"
    ):
        super(RecurrentWrapper, self).__init__()

        self.recurrent_type = recurrent_type

        self.gru_linear = None

        if recurrent_type == "perceiver":
            if type_ == "v1":
                from models.perceiver_wrapper import Perceiver
            elif type_ == "v2":
                from models.perceiver_wrapper import PerceiverV2 as Perceiver
            self.recurrence_model = Perceiver(**perceiver_config)
        elif recurrent_type.lower() == "lstm":
            self.recurrence_model = nn.LSTM(
                input_size=perceiver_config["embedding_dim"],
                hidden_size=perceiver_config["latent_dim"],
                num_layers=perceiver_config["num_tf_layers"],
                proj_size=perceiver_config["output_dim"] if perceiver_config["output_dim"] < perceiver_config["latent_dim"] else 0,
                dropout=dropout_rate,
                batch_first=True
            )
        
        elif recurrent_type.lower() == "gru":
            self.recurrence_model = nn.GRU(
                input_size=perceiver_config["embedding_dim"],
                hidden_size=perceiver_config["latent_dim"],
                num_layers=perceiver_config["num_tf_layers"],
                #proj_size=perceiver_config["output_dim"] if perceiver_config["output_dim"] < perceiver_config["latent_dim"] else 0,
                dropout=dropout_rate,
                batch_first=True
            )
            if perceiver_config["output_dim"] < perceiver_config["latent_dim"]:
                self.gru_linear = nn.Linear(perceiver_config["latent_dim"], perceiver_config["output_dim"])
        else:
            raise ValueError(f"Unsupported recurrent type: {recurrent_type}")

        # Load the DINOv2 model
        self.image_model = AutoModel.from_pretrained("facebook/dinov2-small")
        self.dropout1 = nn.Dropout(dropout_rate)
        self.dropout2 = nn.Dropout(dropout_rate)

        self.freeze_image_model = freeze_image_model
        self.is_append_avg_emb = is_append_avg_emb

    def reset_latents(self):
        if self.recurrent_type == "perceiver":
            self.recurrence_model.reset_latents()

    def image_model_forward(self, video):
        prediction_list = []
        for i in range(video.size(1)):
            frame = video[:,i,:,:,:]
            frame_output = self.image_model(frame).last_hidden_state
            if self.freeze_image_model:
                frame_output = frame_output.detach()
            frame_output = self.dropout2(frame_output)
            prediction_list.append(frame_output)
        return prediction_list

    def get_average(self, emb_list):
        stacked_tensors = torch.stack(emb_list, dim=1)
        stacked_tensors = self.dropout1(stacked_tensors)
        output_tensor = torch.mean(stacked_tensors, dim=-2)
        output_tensor = torch.mean(output_tensor, dim=-2)
        return output_tensor

    def get_max(self, emb_list):
        stacked_tensors = torch.stack(emb_list, dim=1)
        stacked_tensors = self.dropout1(stacked_tensors)
        output_tensor = torch.max(stacked_tensors, dim=-2).values
        output_tensor = torch.max(output_tensor, dim=-2).values
        return output_tensor

    def concat_embeddings(self, emb_list):
        output_tensor = torch.cat(emb_list, dim=1)
        output_tensor = self.dropout1(output_tensor)
        output_tensor = output_tensor.view(emb_list[0].size(0), -1)
        return output_tensor

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
        self.reset_latents()
        batch_size = video.size(0)
        frame_list = []
        for i in range(video.size(1)):
            frame = video[:,i,:,:,:]
            frame_output = self.image_model(frame).last_hidden_state
            if self.freeze_image_model:
                frame_output = frame_output.detach()
            frame_output = self.dropout2(frame_output)
            frame_list.append(frame_output)

        stacked_tensors = torch.stack(frame_list, dim=1)
        avg_image_emb = torch.mean(stacked_tensors, dim=-3)

        if self.recurrent_type == "perceiver":
            prediction_list = []
            for i in range(video.size(1)):
                pred = self.recurrence_model(
                    raw_input=frame_list[i].permute(0, 2, 3, 1) if self.recurrence_model.use_raw_input else None,
                    embeddings=frame_list[i] if self.recurrence_model.use_embeddings else None,
                    video_emb=avg_image_emb if i == 0 else None,
                    is_reset_latents=False
                )
                prediction_list.append(pred)
            self.reset_latents()
            if self.is_append_avg_emb:
                return prediction_list[-1] + self.get_average(frame_list)
            return prediction_list[-1]
        else:
            prediction_list = []
            hidden = None
            for i in range(video.size(1)):
                # Reshape frame_output to be 3D: [batch_size, seq_len, feature_dim]
                #print(f"len(frame_list): {len(frame_list)}")
                #print(frame_list[i].size()) # [batch_size, seq_len, feature_dim]
                
                frame_output = torch.mean(frame_list[i], dim=1)
                #frame_output = frame_output.unsqueeze(1)  # Add sequence dimension
                #print(F"frame_output.size(): {frame_output.size()}")
                output, hidden = self.recurrence_model(frame_output, hidden)
                prediction_list.append(output.squeeze(1))
            if self.gru_linear is not None:
                    prediction_list[-1] = self.gru_linear(prediction_list[-1])
            if self.is_append_avg_emb:
                return prediction_list[-1] + self.get_average(frame_list)
            return prediction_list[-1]

def test_perceiver_wrapper():
    
    #raw_input_dim, embedding_dim, latent_dim, num_heads, num_latents, 
        #num_transformer_layers, dropout, output_dim, use_raw_input=True, use_embeddings=True,
        #flatten_channels=False
    perceiver_config = {
        "raw_input_dim": 3,
        "embedding_dim": 384,
        "latent_dim": 384,
        "num_heads": 8,
        "num_latents": 257,
        "num_transformer_layers": 2,
        "dropout": 0.1,
        "output_dim": 384,
        "use_raw_input": False,
        "use_embeddings": True,
        "flatten_channels": False
    }
    dino_model_name = "facebook/dinov2-small"
    dropout_rate = 0.1
    freeze_image_model = True

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    model = RecurrentWrapper(perceiver_config, dino_model_name, dropout_rate, freeze_image_model, "v2").to(device)
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

def test_lstm():
    perceiver_config = {
        "embedding_dim": 384,
        "latent_dim": 1024,
        "num_tf_layers": 2,
        "output_dim": 384
    }
    dino_model_name = "facebook/dinov2-small"
    dropout_rate = 0.1
    freeze_image_model = True

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    model = RecurrentWrapper(perceiver_config, dino_model_name, dropout_rate, freeze_image_model, recurrent_type="lstm", is_append_avg_emb=True).to(device)
    video = torch.randn(2, 8, 3, 224, 224).to(device)
    output = model(video)
    print(f"(lstm) output.size(): {output.size()}")
    print()

    ## Gradient test
    loss_fn = nn.MSELoss()
    target = torch.randn_like(output)
    loss = loss_fn(output, target)
    model.zero_grad()
    loss.backward()
    gradients = []
    for param in model.parameters():
        if param.grad is not None:
            gradients.append(param.grad.view(-1))
    gradients = torch.cat(gradients)
    print(f"gradients.size(): {gradients.size()}")
    print(f"Gradients: {gradients}")
    print()

def test_gru():
    perceiver_config = {
        "embedding_dim": 384,
        "latent_dim": 1024,
        "num_tf_layers": 2,
        "output_dim": 384
    }
    dino_model_name = "facebook/dinov2-small"
    dropout_rate = 0.1
    freeze_image_model = True

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    model = RecurrentWrapper(perceiver_config, dino_model_name, dropout_rate, freeze_image_model, recurrent_type="gru", is_append_avg_emb=True).to(device)
    video = torch.randn(2, 8, 3, 224, 224).to(device)
    output = model(video)
    print(f"(gru) output.size(): {output.size()}")
    print()

    ## Gradient test
    loss_fn = nn.MSELoss()
    target = torch.randn_like(output)
    loss = loss_fn(output, target)
    model.zero_grad()
    loss.backward()
    gradients = []
    for param in model.parameters():
        if param.grad is not None:
            gradients.append(param.grad.view(-1))
    gradients = torch.cat(gradients)
    print(f"gradients.size(): {gradients.size()}")
    print(f"Gradients: {gradients}")
    print()

if __name__ == "__main__":
    
    print(f"Perceiver wrapper:\n\n")
    test_perceiver_wrapper()
    print(f"\n\nLSTM:\n\n")
    test_lstm()
    print(f"\n\nGRU:\n\n")
    test_gru()

