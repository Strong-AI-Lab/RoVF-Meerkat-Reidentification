import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from transformers import AutoModel

class RecurrentDecoder(nn.Module):
    
    def __init__(
        self, v_size, d_model, nhead, num_layers, dim_feedforward, dropout, activation='gelu', 
        temperature=1.0, image_model_name="facebook/dinov2-small", freeze_image_model=True,
        image_model_sl=257
    ):
        super(RecurrentDecoder, self).__init__()
        
        self.v_size = v_size
        self.d_model = d_model
        self.nhead = nhead
        self.num_layers = num_layers
        self.dim_feedforward = dim_feedforward
        self.dropout = dropout
        self.activation = activation
        self.temperature = temperature
        self.eval_mode = False  # New attribute for evaluation mode

        # Image model
        self.image_model = AutoModel.from_pretrained(image_model_name)
        if freeze_image_model:
            for param in self.image_model.parameters():
                param.requires_grad = False
        self.freeze_image_model = freeze_image_model

        # Transformer decoder layer
        dec_layer = nn.TransformerDecoderLayer(d_model, nhead, dim_feedforward, dropout, activation, batch_first=True)
        self.decoder = nn.TransformerDecoder(dec_layer, num_layers)

        # Random vocabulary embeddings
        embedding_dim = d_model  # Assuming vocab embedding dimension matches d_model
        self.vocab_embeddings = nn.Parameter(nn.init.xavier_uniform_(torch.empty(v_size, embedding_dim)))

        # Learnable initial latent array
        self.latents = nn.Parameter(nn.init.xavier_uniform_(torch.empty(1, v_size, d_model)))  # Initialized latent tensor

        # Output linear layer
        self.linear_out = nn.Linear(d_model, v_size)
        self.linear_emb = nn.Linear(d_model*image_model_sl, embedding_dim)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def set_eval_mode(self, mode):
        self.eval_mode = mode

    def softmax_with_temperature(self, logits):
        # Apply softmax with temperature
        softmax_output = F.softmax(logits / self.temperature, dim=-1)
        return softmax_output

    def weighted_sum_vocabulary(self, softmax_output):
        if self.eval_mode:
            # In evaluation mode, select the maximum embedding
            max_indices = torch.argmax(softmax_output, dim=-1)
            return self.vocab_embeddings[max_indices]
        else:
            # In training mode, compute weighted sum of vocab embeddings
            return torch.matmul(softmax_output, self.vocab_embeddings)

    def forward(self, video):
        # video.size() = (batch_size, #frames, 3, 224, 224)
        
        batch_size = video.size(0)
        latent = self.latents.expand(batch_size, -1, -1)  # Expand latents to match batch size
        latent = self.dropout1(latent)

        for i in range(video.size(1)):
            frame = video[:, i, :, :, :]
            frame = self.image_model(frame).last_hidden_state  # Use the image model's last hidden state
            #print(f"frame.size(): {frame.size()}")
            # Generate a target mask for the decoder
            tgt_mask = torch.nn.Transformer.generate_square_subsequent_mask(frame.size(1)).to(frame.device)

            # Compute output via the transformer decoder
            output = self.decoder(tgt=frame, memory=latent, tgt_mask=tgt_mask)  # (batch_size, slen, d_model)
            output_vocab = self.linear_out(output)  # (batch_size, slen, v_size)

            # Apply dropout
            output_vocab = self.dropout2(output_vocab)

            # Apply softmax and temperature scaling
            vocab_probs = self.softmax_with_temperature(output_vocab)  # (batch_size, slen, v_size)

            # Update latent with weighted sum of vocab embeddings (or max embedding in eval mode)
            latent = self.weighted_sum_vocabulary(vocab_probs)  # (batch_size, slen, d_model)

        return self.linear_emb(latent.view(latent.size(0), -1))  # Return the embeddings

if __name__ == "__main__":
    # Initialize the recurrent decoder model
    model = RecurrentDecoder(v_size=128, d_model=384, nhead=8, num_layers=2, dim_feedforward=384*4, dropout=0.1, freeze_image_model=False)

    # Generate random video input
    batch_size = 2
    sequence_length = 10
    video = torch.randn(batch_size, sequence_length, 3, 224, 224)  # (batch_size, sequence_length, channels, height, width)

    output = model(video)  # Forward pass
    print(output.size())  # Expected output: torch.Size([2, 10, 384])

    output.sum().backward()  # Perform backward pass

    # go through all parameters and check if they have gradients; print out name if they have no gradients
    for name, param in model.named_parameters():
        if not param.grad is None:
            #print(name, param.grad.sum())
            assert not torch.isnan(param.grad.sum()), f"NaN found in gradients of {name}"
        else:
            print(name, "has no gradients")

    # test out the set_eval_mode function
    model.set_eval_mode(True)

    out = model(video)  # Forward pass in evaluation mode
    print("Forward pass in evaluation mode successful.")
    print(f"Output size: {out.size()}")  # Expected output: torch.Size([2, 10, 384])


