import torch
from transformers import AutoModel

import torch.nn as nn

class ViViTWrapper(nn.Module):
    def __init__(self, vivit_model_name, output_dim, dropout_rate=0.1):
        super(ViViTWrapper, self).__init__()
        
        # Load the ViViT model
        self.vivit = AutoModel.from_pretrained(vivit_model_name)

        # Get the dimension of the output features from ViViT
        self.vivit_output_dim = self.vivit.config.hidden_size

        if output_dim is not None:
            
            self.vivit_output_dim
            self.linear = nn.Linear(
                self.vivit_output_dim, output_dim
            )
        else:
            self.linear = None

        self.dropout1 = nn.Dropout(dropout_rate)
        self.dropout2 = nn.Dropout(dropout_rate)

    def forward(self, video):
        # video.size() = batch, #frames, #channels, height, width
        assert len(video.size()) == 5, f"video.size(): {video.size()}; expected 5 dimensions (batch, #frames, #channels, height, width)."
        
        cls_outputs = self.vivit(video).last_hidden_state  # (batch, num_frames, hidden_size)
        
        output_tensor = cls_outputs[:, 0, :]  # (batch, hidden_size)
        
        output_tensor = self.dropout2(output_tensor)
        if self.linear is not None:
            linear_output = self.linear(output_tensor)
        else:
            linear_output = output_tensor

        return linear_output

def forward_test(output_dim):
    print(f"Test with output_dim={output_dim}")

    vivit_model_name = 'google/vivit-b-16x2-kinetics400'

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    batch_size = 4

    model = ViViTWrapper(
        vivit_model_name, output_dim, dropout_rate=0.0
    )
    model.to(device)

    video = torch.randn(batch_size, 32, 3, 224, 224).to(device)  # Assuming 16 frames per video

    # Define a dummy target and loss function
    if output_dim is not None:
        target = torch.randn(batch_size, output_dim).to(device)  # Assuming output_dim matches the target size
    else:
        target = torch.randn(batch_size, model.vivit_output_dim).to(device)

    criterion = nn.MSELoss()

    # Forward pass
    output = model(video)
    print(f"output.size(): {output.size()}")

    # Compute loss
    loss = criterion(output, target)
    print(f"Loss: {loss.item()}")

    # Backward pass to compute gradients
    loss.backward()

    # Check gradients
    for name, param in model.named_parameters():
        if param.grad is not None:
            print(f"Gradient for {name}: {param.grad.norm().item()}")
        else:
            print(f"No gradient computed for {name}")

def print_model_architecture():
    model = ViViTWrapper('google/vivit-b-16x2-kinetics400', 1000)
    print(model)

if __name__ == "__main__":
    forward_test(1000) # b, 1000
    forward_test(None) # b, d_m (768)

    print_model_architecture()