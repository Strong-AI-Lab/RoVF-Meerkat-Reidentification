import sys
sys.path.append("..")

import torch
import torch.nn as nn
import torch.nn.functional as F

class CrossAttention(nn.Module):
    def __init__(self, latent_dim, data_dim, num_heads):
        super().__init__()
        self.data_dim = data_dim
        self.query_proj = nn.Linear(latent_dim, latent_dim)
        self.key_proj = nn.Linear(data_dim, latent_dim)
        self.value_proj = nn.Linear(data_dim, latent_dim)
        self.num_heads = num_heads
        # TODO: at some point this should be flash attention.
        self.attention = nn.MultiheadAttention(latent_dim, num_heads, batch_first=True)

    def forward(self, latents, data):
        # Ensure data is correctly shaped for multi-head attention
        query = self.query_proj(latents)
        
        if len(data.size()) == 2:
            #assumes (b, slen)
            data = data.unsqueeze(1)
            #data = data.view(data.size(0), -1, self.data_dim)
        
        #print(f"data.size(): {data.size()}")
            
        key = self.key_proj(data)
        value = self.value_proj(data)

        #print()
        #print(f"query.size(): {query.size()}")
        #print(f"key.size(): {key.size()}")
        #print(f"value.size(): {value.size()}")
        #print()

        # Apply multi-head attention
        attn_output, _ = self.attention(
            query,
            key,
            value
        )
        return attn_output

class TransformerEncoder(nn.Module):
    def __init__(self, latent_dim, num_heads, num_layers, dropout):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(d_model=latent_dim, nhead=num_heads, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, src):
        return self.transformer_encoder(src)

class Perceiver(nn.Module):
    def __init__(
        self, input_dim, latent_dim, num_heads, num_latents, 
        num_transformer_layers, dropout, output_dim
    ):
        super().__init__()
        
        self.latents_p = nn.Parameter(torch.randn(num_latents, latent_dim))
        nn.init.xavier_uniform_(self.latents_p)
        self.latents = None # store recurrent modification of initial value.
        
        # Initialize positional embeddings for latents
        self.positional_embeddings = nn.Parameter(torch.randn(num_latents, latent_dim))
        nn.init.xavier_uniform_(self.positional_embeddings)

        #self.latents = nn.Parameter(torch.zeros(num_latents, latent_dim))
        
        self.latent_batch_dimension_set = False

        # Ensure the key and value projections in CrossAttention are compatible with the input dimension.
        self.cross_attention = CrossAttention(latent_dim, input_dim, num_heads) # Adjusted to project input data correctly
        self.transformer = TransformerEncoder(latent_dim, num_heads, num_transformer_layers, dropout)

        if output_dim is None:
            self.output_layer = None
        else:
            self.output_layer = nn.Linear(num_latents*latent_dim, output_dim)

    def set_latent_batch_dimension(self, batch_size, latents):
        #if not self.latent_batch_dimension_set:
        latents = latents.unsqueeze(0).repeat(batch_size, 1, 1)
        #self.latent_batch_dimension_set = True
        return latents

    def reset_latents(self):
        self.latents = None
        #self.latent_batch_dimension_set = False

    def forward(self, x, is_reset_latents=False):
        # x.size() == (batch_size, seq_len, input_dim)
        
        assert len(x.size()) == 3, f"x.size(): {x.size()}; expected 3 dimensions."

        batch_size = x.size(0)

        # set latents to self.latents_p and add batch dimension.
        if self.latents is None:
            latents = self.latents_p
            latents = latents.unsqueeze(0).repeat(batch_size, 1, 1)
        else:
            latents = self.latents

        # Add positional embeddings to latents
        latents = latents + self.positional_embeddings.unsqueeze(0).repeat(batch_size, 1, 1)

        x = self.cross_attention(latents, x)
        x = self.transformer(x)

        latents = x

        if self.output_layer is not None:
            x = self.output_layer(x.view(x.size(0), -1))

        if is_reset_latents:
            self.reset_latents()
        else:
            self.latents = latents # store latents for the next forward pass (for an outside recurrent wrapper).
        return x

def cross_attention_test():
    latent_dim = 32
    data_dim = 64
    num_heads=2
    latents = torch.randn(2, 2, latent_dim, requires_grad=True)
    data = torch.randn(2, 2, data_dim, requires_grad=True)

    model = CrossAttention(latent_dim, data_dim, num_heads)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    criterion = torch.nn.MSELoss()

    output = model(latents, data)

    # Create a simple target
    target = torch.randn_like(output)
    loss = criterion(output, target)

    optimizer.zero_grad()
    loss.backward()

    print(f"Gradient for query_proj.weight: {model.query_proj.weight.grad}")
    print(f"Gradient for key_proj.weight: {model.key_proj.weight.grad}")
    print(f"Gradient for value_proj.weight: {model.value_proj.weight.grad}")

    optimizer.step()

def perceiver_test():
    perceiver = Perceiver(
        input_dim=10, latent_dim=32, num_heads=1, num_latents=64, 
        num_transformer_layers=2, dropout=0.1, output_dim=10
    )
    
    # Example input
    x = torch.randn(5, 10, 10)
    
    # Forward pass
    latents = perceiver(x, is_reset_latents=True)
    print(f"latents.size(): {latents.size()}")

    # Example target
    target = torch.randn_like(latents)
    
    # Define a loss function (e.g., mean squared error)
    loss_fn = nn.MSELoss()
    
    # Calculate loss
    loss = loss_fn(latents, target)
    print(f"Loss: {loss.item()}")

    # Backpropagation
    loss.backward()

    # Check if gradients exist for latents_p
    print(f"Gradients on latents_p: {perceiver.latents_p.grad}")



if __name__ == "__main__":
    
    #perceiver_test()
    cross_attention_test()
