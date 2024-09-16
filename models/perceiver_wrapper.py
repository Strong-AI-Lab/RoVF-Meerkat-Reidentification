import sys
sys.path.append("..")

import torch
import torch.nn as nn
import torch.nn.functional as F

import transformers 
from transformers import AutoModel



class TransformerEncoder(nn.Module):
    def __init__(self, latent_dim, num_heads, num_layers, dropout):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=latent_dim, nhead=num_heads, dim_feedforward=latent_dim*4, 
            dropout=dropout, activation='gelu', batch_first=True, norm_first=True
        )
        norm = nn.LayerNorm(normalized_shape=latent_dim)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers, norm=norm)

    def forward(self, src, mask=None):
        return self.transformer_encoder(src, mask=mask)

class TransformerDecoder(nn.Module):
    def __init__(self, latent_dim, num_heads, num_layers, dropout):
        super().__init__()
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=latent_dim, nhead=num_heads, dim_feedforward=latent_dim*4, 
            dropout=dropout, activation='gelu', batch_first=True
        )
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)

    def forward(self, src):
        return self.transformer_decoder(src)

class CrossAttention(nn.Module):
    def __init__(self, latent_dim, data_dim, num_heads):
        super().__init__()
        self.data_dim = data_dim
        self.query_proj = nn.Linear(latent_dim, latent_dim)
        self.key_proj = nn.Linear(data_dim, latent_dim)
        self.value_proj = nn.Linear(data_dim, latent_dim)
        self.num_heads = num_heads
        self.attention = nn.MultiheadAttention(latent_dim, num_heads, batch_first=True)

    def forward(self, latents, data):
        query = self.query_proj(latents)
        
        if len(data.size()) == 2:
            data = data.view(data.size(0), -1, self.data_dim)
        
        key = self.key_proj(data)
        value = self.value_proj(data)

        attn_output, _ = self.attention(query, key, value)
        return attn_output

#import torch
#import torch.nn as nn
#from transformers import AutoModel, AutoTokenizer
'''
class Perceiver(nn.Module):
    def __init__(
        self, raw_input_dim, embedding_dim, latent_dim, num_heads, num_latents, 
        num_transformer_layers, dropout, output_dim, use_raw_input=True, use_embeddings=True,
        flatten_channels=False
    ):
        super().__init__()
        
        self.latents_p = nn.init.xavier_uniform_(nn.Parameter(torch.randn(num_latents, latent_dim)))
        self.latents = None
        
        self.num_latents = num_latents
        self.raw_input_dim = raw_input_dim
        self.embedding_dim = embedding_dim
        self.use_raw_input = use_raw_input
        self.use_embeddings = use_embeddings
        self.flatten_channels = flatten_channels

        if use_raw_input:
            self.raw_cross_attention = CrossAttention(latent_dim, 1 if flatten_channels else raw_input_dim, num_heads)
        if use_embeddings:
            self.embedding_cross_attention = CrossAttention(latent_dim, embedding_dim, num_heads)

        # Load RoBERTa model
        self.transformer = AutoModel.from_pretrained("roberta-large")

        if output_dim is not None:
            self.output_layer = nn.Linear(num_latents * latent_dim, output_dim)
            self.layer_norm3 = nn.LayerNorm(output_dim)
        else:
            self.output_layer = None
            self.layer_norm3 = nn.LayerNorm(num_latents * latent_dim)

        self.layer_norm1 = nn.LayerNorm(embedding_dim)
        self.layer_norm2 = nn.LayerNorm(latent_dim)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.tokenizer = AutoTokenizer.from_pretrained("roberta-large")

    def reset_latents(self):
        self.latents = None

    def forward(self, raw_input=None, embeddings=None, is_reset_latents=False):
        if not self.use_raw_input and not self.use_embeddings:
            raise ValueError("At least one of use_raw_input or use_embeddings must be True")

        if self.use_raw_input and raw_input is None:
            raise ValueError("raw_input is required when use_raw_input is True")

        if self.use_embeddings and embeddings is None:
            raise ValueError("embeddings is required when use_embeddings is True")

        batch_size = raw_input.size(0) if raw_input is not None else embeddings.size(0)

        if self.latents is None:
            latents = self.latents_p.unsqueeze(0).repeat(batch_size, 1, 1)
            latents = self.layer_norm2(latents)
        else:
            latents = self.latents

        latents = self.dropout1(latents)

        if self.use_raw_input:
            raise Exception("Not tested with raw input!")
            if self.flatten_channels:
                flattened_raw_input = raw_input.reshape(raw_input.size(0), -1, 1)
            else:
                flattened_raw_input = raw_input.reshape(raw_input.size(0), -1, self.raw_input_dim)
            latents = self.raw_cross_attention(latents, flattened_raw_input)
            latents = self.dropout2(latents)

        if self.use_embeddings:
            #embeddings = self.layer_norm1(embeddings)
            latents = self.embedding_cross_attention(latents, embeddings)
            latents = self.dropout3(latents)
        #print(f"latents.size(): {latents.size()}")

        # Handle tokenization and attention masks for RoBERTa
        if self.use_raw_input:
            raise Exception("Not tested with raw input!")
            raw_input_tokens = [self.tokenizer.encode(text, return_tensors='pt') for text in raw_input]
            attention_masks = [torch.ones(tokens.size()) for tokens in raw_input_tokens]
            raw_input_tokens = torch.cat(raw_input_tokens, dim=0)
            attention_masks = torch.cat(attention_masks, dim=0)
            
            roberta_outputs = self.transformer(input_ids=raw_input_tokens, attention_mask=attention_masks)
            roberta_hidden_states = roberta_outputs.last_hidden_state

            # Aggregate hidden states (e.g., take the mean or [CLS] token representation)
            roberta_representation = roberta_hidden_states.mean(dim=1)  # or use [CLS] token

            # Integrate with latents
            latents = latents + roberta_representation.unsqueeze(1)
            latents = self.dropout3(latents)

        # RoBERTa processing with input embeddings
        roberta_outputs = self.transformer(inputs_embeds=latents, attention_mask=None)
        roberta_hidden_states = roberta_outputs.last_hidden_state

        # Aggregate hidden states (e.g., mean or [CLS] token representation)
        #roberta_representation = roberta_hidden_states.mean(dim=1)  # or use [CLS] token

        # Integrate with latents
        # TODO consider below
        #print(f"roberta_hidden_states.size(): {roberta_hidden_states.size()}")
        #latents = latents + roberta_hidden_states #roberta_representation.unsqueeze(1)
        latents = roberta_hidden_states
        latents = self.dropout3(latents)
        #print(f"latents.size(): {latents.size()}")

        if self.output_layer is not None:
            output = self.output_layer(latents.view(latents.size(0), -1))
        else:
            output = latents.view(latents.size(0), -1)

        if is_reset_latents:
            self.reset_latents()
        else:
            self.latents = latents

        return output

def test_perceiver_lm():
    # Test parameters
    raw_input_dim = 768  # Example dimension for raw input
    embedding_dim = 384  # Example dimension for embeddings
    latent_dim = 1024
    num_heads = 8
    num_latents = 10
    num_transformer_layers = 6
    dropout = 0.1
    output_dim = 5  # Example output dimension
    batch_size = 2
    seq_len = 20  # Example sequence length for raw input/embeddings

    # Instantiate the Perceiver model
    model = Perceiver(
        raw_input_dim=raw_input_dim,
        embedding_dim=embedding_dim,
        latent_dim=latent_dim,
        num_heads=num_heads,
        num_latents=num_latents,
        num_transformer_layers=num_transformer_layers,
        dropout=dropout,
        output_dim=output_dim,
        use_raw_input=False,
        use_embeddings=True
    )

    # Create dummy raw input and embeddings
    raw_input = None #torch.randn(batch_size, seq_len, raw_input_dim)
    embeddings = torch.randn(batch_size, seq_len, embedding_dim)

    # Forward pass with both raw input and embeddings
    output = model(embeddings=embeddings)
'''
# Run the test
#test_perceiver()

 

class Perceiver(nn.Module):
    def __init__(
        self, raw_input_dim, embedding_dim, latent_dim, num_heads, num_latents, 
        num_transformer_layers, dropout, output_dim, use_raw_input=True, use_embeddings=True,
        flatten_channels=False, use_video_emb=False
    ):
        super().__init__()

        self.latents_p = nn.Parameter(nn.init.xavier_uniform_(torch.randn(num_latents, latent_dim)) * (latent_dim ** 0.5))
        self.latents = None
        self.video_emb = None
        
        self.use_video_emb = use_video_emb
        if use_video_emb:
            pe_nlatents = num_latents + 1
        else:
            pe_nlatents = num_latents
        self.positional_embeddings = nn.init.xavier_uniform_(nn.Parameter(torch.randn(pe_nlatents, latent_dim)))
        
        self.num_latents = num_latents
        self.raw_input_dim = raw_input_dim
        self.embedding_dim = embedding_dim
        self.use_raw_input = use_raw_input
        self.use_embeddings = use_embeddings
        assert (use_raw_input or use_embeddings) and not (use_raw_input and use_embeddings), "At least one of use_raw_input or use_embeddings must be True"
        self.flatten_channels = flatten_channels

        if use_raw_input:
            self.raw_cross_attention = CrossAttention(latent_dim, embedding_dim, num_heads)
        if use_embeddings:
            self.embedding_cross_attention = CrossAttention(latent_dim, embedding_dim, num_heads)


        self.transformer = TransformerEncoder(latent_dim, num_heads, num_transformer_layers, dropout)


        if output_dim is not None:
            self.output_layer = nn.Linear(latent_dim*num_latents, output_dim)
        else:
            self.output_layer = None

        self.layer_norm1 = nn.LayerNorm(embedding_dim)
        self.layer_norm2 = nn.LayerNorm(latent_dim)


        self.dropout1 = nn.Dropout(dropout)        
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

    def reset_latents(self):
        self.latents = None
        self.video_emb = None


    def forward(self, raw_input=None, embeddings=None, video_emb=None, is_reset_latents=False):
        
        if not self.use_raw_input and not self.use_embeddings:
            raise ValueError("At least one of use_raw_input or use_embeddings must be True")

        if self.use_raw_input and raw_input is None:
            raise ValueError("raw_input is required when use_raw_input is True")
        # raw_input: (batch_size, h, w, c)
        # the channel is expected to be at the end. 

        if self.use_embeddings and embeddings is None:
            raise ValueError("embeddings is required when use_embeddings is True")

        batch_size = raw_input.size(0) if raw_input is not None else embeddings.size(0)

        latents = None
        if self.latents is None:
            if video_emb is not None:
                latents = self.layer_norm2(video_emb)
            else:
                latents = self.latents_p.unsqueeze(0).repeat(batch_size, 1, 1)
        else:
            latents = self.latents
        
        #if add_pos_emb:
        #    latents = latents + self.positional_embeddings.unsqueeze(0).repeat(batch_size, 1, 1)

        latents = self.dropout1(latents)

        if self.use_raw_input:
            flattened_raw_input = raw_input.view(raw_input.size(0), -1, raw_input.size(-1)) # already normalized
            latents = self.raw_cross_attention(latents, flattened_raw_input) + latents # residual connection
            latents = self.dropout2(latents)
        if self.use_embeddings:
            embeddings = self.layer_norm1(embeddings)
            latents = self.embedding_cross_attention(latents, embeddings) + latents # residual connection
            latents = self.dropout3(latents)

        latents_res = latents  # Save the original latents for residual connection
        latents = self.transformer(latents)
        latents = latents + latents_res  # Residual connection

        if self.output_layer is not None:
            output = self.output_layer(latents.view(latents.size(0), -1))
        else:
            output = latents[:,0,:]

        if is_reset_latents:
            self.reset_latents()
        else:
            self.latents = latents

        return output

class PerceiverV2(nn.Module):
    def __init__(
        self, raw_input_dim, embedding_dim, latent_dim, num_heads, num_latents, 
        num_transformer_layers, dropout, output_dim, use_raw_input=True, use_embeddings=True,
        flatten_channels=False, use_video_emb=False
    ):
        super().__init__()
        #print(F"\n\n\n REACH \n\n\n")
        self.latents_p = nn.Parameter(nn.init.xavier_uniform_(torch.randn(num_latents, latent_dim)) * (latent_dim ** 0.5))
        self.latents = None
        
        self.use_video_emb = use_video_emb
        if use_video_emb:
            pe_nlatents = num_latents + 1
        else:
            pe_nlatents = num_latents
        self.positional_embeddings = nn.init.xavier_uniform_(nn.Parameter(torch.randn(pe_nlatents, latent_dim)))
        
        self.num_latents = num_latents
        self.raw_input_dim = raw_input_dim
        self.embedding_dim = embedding_dim
        self.use_raw_input = use_raw_input
        self.use_embeddings = use_embeddings
        assert (use_raw_input or use_embeddings) and not (use_raw_input and use_embeddings), "At least one of use_raw_input or use_embeddings must be True"
        self.flatten_channels = flatten_channels

        if use_raw_input:
            self.raw_cross_attention = CrossAttention(latent_dim, embedding_dim, num_heads)
        if use_embeddings:
            self.embedding_cross_attention = CrossAttention(latent_dim, embedding_dim, num_heads)

        self.transformer = TransformerEncoder(latent_dim, num_heads, num_transformer_layers, dropout)

        if output_dim is not None:
            self.output_layer = nn.Linear(latent_dim*num_latents, output_dim)
        else:
            self.output_layer = None

        self.layer_norm1 = nn.LayerNorm(embedding_dim)
        self.layer_norm2 = nn.LayerNorm(latent_dim)

        self.dropout1 = nn.Dropout(dropout)        
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

    def reset_latents(self):
        self.latents = None

    def forward(self, raw_input=None, embeddings=None, video_emb=None, add_pos_emb=True, is_reset_latents=False):
        if not self.use_raw_input and not self.use_embeddings:
            raise ValueError("At least one of use_raw_input or use_embeddings must be True")

        if self.use_raw_input and raw_input is None:
            raise ValueError("raw_input is required when use_raw_input is True")

        if self.use_embeddings and embeddings is None:
            raise ValueError("embeddings is required when use_embeddings is True")

        batch_size = raw_input.size(0) if raw_input is not None else embeddings.size(0)

        if self.latents is None:
            latents = self.latents_p.unsqueeze(0).repeat(batch_size, 1, 1)
        else:
            latents = self.latents
        
        if add_pos_emb:
            latents = latents + self.positional_embeddings.unsqueeze(0).repeat(batch_size, 1, 1)

        latents = self.dropout1(latents)

        if self.use_raw_input:
            flattened_raw_input = raw_input.view(raw_input.size(0), -1, raw_input.size(-1))
            latents = self.raw_cross_attention(latents, flattened_raw_input) + latents
            latents = self.dropout2(latents)
        if self.use_embeddings:
            embeddings = self.layer_norm1(embeddings)
            latents = self.embedding_cross_attention(latents, embeddings) + latents
            latents = self.dropout3(latents)

        latents = self.transformer(latents)

        if self.output_layer is not None:
            output = self.output_layer(latents.view(latents.size(0), -1))
        else:
            output = latents[:,0,:]

        if is_reset_latents:
            self.reset_latents()
        else:
            self.latents = latents

        return output

def cross_attention_test():
    latent_dim = 32
    data_dim = 1
    num_heads=2
    latents = torch.randn(2, 50, latent_dim, requires_grad=True)
    data = torch.randn(2, 2, requires_grad=True)

    model = CrossAttention(latent_dim, data_dim, num_heads)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    criterion = torch.nn.MSELoss()

    output = model(latents, data)
    print(f"output.size(): {output.size()}")
    # Create a simple target
    target = torch.randn_like(output)
    loss = criterion(output, target)

    optimizer.zero_grad()
    loss.backward()

    #print(f"Gradient for query_proj.weight: {model.query_proj.weight.grad}")
    #print(f"Gradient for key_proj.weight: {model.key_proj.weight.grad}")
    #print(f"Gradient for value_proj.weight: {model.value_proj.weight.grad}")

    optimizer.step()

def perceiver_test():
    perceiver = Perceiver(
        input_dim=32, latent_dim=32, num_heads=1, num_latents=64, 
        num_transformer_layers=2, dropout=0.1, output_dim=10
    )
    
    # Example input
    x = torch.randn(5, 122, 32)
    
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

def upd_perc_test():
    perceiver = Perceiver(
        raw_input_dim=3,  # For RGB images
        embedding_dim=512,  # Example embedding dimension
        latent_dim=256,
        num_heads=8,
        num_latents=64,
        num_transformer_layers=6,
        dropout=0.1,
        output_dim=10,
        use_raw_input=True,
        use_embeddings=True,
        flatten_channels=False  # or True, depending on your preference
    )

    # Example usage
    raw_input = torch.randn(32, 224, 224, 3)  # Batch of 32 RGB images
    embeddings = torch.randn(32, 512)  # Batch of 32 image embeddings

    output = perceiver(raw_input=raw_input, embeddings=embeddings)
    print(f"output.size(): {output.size()}")

if __name__ == "__main__":
    
    #perceiver_test()
    #cross_attention_test()
    #upd_perc_test()
    test_perceiver_lm()
