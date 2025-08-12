import torch
import torch.nn as nn
import open_clip
import math

class BioCLIPVideoWrapper(nn.Module):
    def __init__(self, model_name, output_dim, forward_strat: str="cat", sequence_length=None, num_frames: int=1, dropout_rate=0.1, checkpoint_path=None,
                 return_prepool: bool=False, prepool_return_grid: bool=False, prepool_only_patches: bool=False):
        super(BioCLIPVideoWrapper, self).__init__()
        
        # Load the BioCLIP model
        self.model, self.preprocess_train, self.preprocess_val = open_clip.create_model_and_transforms(model_name)
        self.tokenizer = open_clip.get_tokenizer(model_name)
        if checkpoint_path:
            self.model.load_state_dict(torch.load(checkpoint_path))

        # New pre-pool config
        self.return_prepool = return_prepool
        self.prepool_return_grid = prepool_return_grid
        self.prepool_only_patches = prepool_only_patches

        self.forward_strat = forward_strat
        if self.forward_strat not in ["cat", "average", "avg", "mean", "max", "maximum", "cls"]:
            raise ValueError(f"Invalid forward strategy: {self.forward_strat}. Please use one of 'cat', 'average', or 'max'.")
        self.sequence_length = sequence_length
        self.num_frames = num_frames
        image = torch.randn(1, 3, 224, 224) # (b, c, h, w)
        self._compute_sequence_length(image)
        if self.sequence_length is None and forward_strat == "cat":
            self.sequence_length = self._compute_sequence_length(image)
            self.sequence_length = self.sequence_length * self.num_frames
        
        if self.forward_strat == "cat":
            assert self.sequence_length is not None, f"Sequence length must be provided when using the 'cat' forward strategy. Got {self.sequence_length}"

        # Get the dimension of the output features from BioCLIP
        self.model_output_dim = self.model.visual.output_dim

        if output_dim is not None:
            if self.forward_strat == "cat":
                x = self.model_output_dim * self.num_frames
            else:
                x = self.model_output_dim
            self.linear = nn.Linear(x, output_dim)
        else:
            self.linear = None

        self.dropout1 = nn.Dropout(dropout_rate)
        self.dropout2 = nn.Dropout(dropout_rate)

    def _compute_sequence_length(self, image):
        size_ = self.model.encode_image(image).size()
        assert len(size_) == 2, f"The output of the BioCLIP model is not of the expected shape. Expected 2 dimensions, got {len(size_)}"
        return size_[1]

    def forward(self, video):
        if len(video.size()) == 4: # assume that this is because num_frames=1
            video = video.unsqueeze(1)
        assert len(video.size()) == 5, f"video.size(): {video.size()}; expected 5 dimensions (batch, #frames, #channels, height, width)."

        # If requested, return pre-pooling tokens from the image encoder (no temporal aggregation/linear)
        if self.return_prepool:
            return self.extract_prepool_tokens(video, return_grid=self.prepool_return_grid, only_patches=self.prepool_only_patches)

        cls_outputs = [self.model.encode_image(video[:,i,:,:,:]) for i in range(video.size(1))] # [#frames] [b, dm]
        #print(cls_outputs[0].size()) # b, dm
        num_frames = len(cls_outputs)

        batch_size = video.size(0)

        assert len(cls_outputs[0].size()) == 2, f"The output of the BioCLIP model is not of the expected shape. Expected 2 dimensions, got {len(cls_outputs[0].size())}"
        
        if self.forward_strat == "cat": 
            output_tensor = torch.stack(cls_outputs, dim=1) # (b, #frames, dm)
            output_tensor = output_tensor.view(batch_size, -1) #(b, #frames * dm)
            output_tensor = self.dropout1(output_tensor) 
        elif self.forward_strat in ["average", "avg", "mean"]:
            stacked_tensors = torch.stack(cls_outputs, dim=1) # (b, #frames, dm)
            output_tensor = torch.mean(stacked_tensors, dim=1) # average along frame dimension
        elif self.forward_strat in ["max", "maximum"]:
            stacked_tensors = torch.stack(cls_outputs, dim=1) # (b, #frames, dm)
            output_tensor = torch.max(stacked_tensors, dim=1).values # max along frame dimension
        elif self.forward_strat == "cls":
            output_tensor = cls_outputs[-1] # (b, dm) # if this is a video, take the last frame. Assume in practice that only one frame is provided. 
            # this clip model already outputs a single dm vector. So no need to do anything else.
        else:    
            raise ValueError(f"Invalid forward strategy: {self.forward_strat}. Please use one of 'cat', 'average', or 'max'.")
        
        output_tensor = self.dropout2(output_tensor)
        if self.linear is not None:
            linear_output = self.linear(output_tensor)
        else:
            linear_output = output_tensor

        return linear_output

    # --- New utility: extract token embeddings before pooling/projection ---
    #@torch.no_grad()
    def extract_prepool_tokens(self, video, return_grid: bool=False, only_patches: bool=False):
        """Return transformer token embeddings before pooling/projection.
        Args:
            video: (B,T,C,H,W) or (B,C,H,W)
            return_grid: if True and patch tokens form a square grid, returns (B,T,C,H,W) using patch tokens.
            only_patches: if True, drop the CLS token from the returned tokens.
        Returns:
            If return_grid: FloatTensor (B,T,C,H,W)
            Else: FloatTensor (B,T,L,C) where L = 1+N (or N if only_patches)
        """
        was_4d = False
        if video.dim() == 4:
            video = video.unsqueeze(1)
            was_4d = True
        assert video.dim() == 5, f"Expected video of shape (B,T,C,H,W) or (B,C,H,W), got {tuple(video.size())}"

        B, T, C, H, W = video.size()
        tokens_per_frame = []

        def _hook(module, inp, out):
            # capture the sequence of tokens output by the transformer
            self._prepool_tokens_tmp = out

        handle = None
        self._prepool_tokens_tmp = None
        try:
            # Prefer hooking the transformer output (sequence of tokens)
            if hasattr(self.model.visual, 'transformer') and isinstance(self.model.visual.transformer, nn.Module):
                handle = self.model.visual.transformer.register_forward_hook(_hook)
            else:
                # Fallback: try ln_post (may capture CLS-only in some builds)
                if hasattr(self.model.visual, 'ln_post') and isinstance(self.model.visual.ln_post, nn.Module):
                    handle = self.model.visual.ln_post.register_forward_hook(_hook)
                else:
                    raise RuntimeError("Could not find a module to hook for pre-pooling tokens in BioCLIP visual model.")

            for i in range(T):
                _ = self.model.encode_image(video[:, i])  # triggers hooks
                x = self._prepool_tokens_tmp
                if x is None:
                    raise RuntimeError("Failed to capture pre-pooling tokens from BioCLIP.")
                # Normalize to (B, L, C)
                if x.dim() == 3:
                    if x.size(0) == B:  # (B,L,C)
                        seq_tokens = x
                    elif x.size(1) == B:  # (L,B,C) -> (B,L,C)
                        seq_tokens = x.permute(1, 0, 2).contiguous()
                    else:
                        raise RuntimeError(f"Unexpected token shape from hook: {tuple(x.size())}")
                elif x.dim() == 2:
                    seq_tokens = x.unsqueeze(1)  # (B,1,C)
                else:
                    raise RuntimeError(f"Unexpected token shape from hook: {tuple(x.size())}")

                if only_patches:
                    if seq_tokens.size(1) > 1:
                        seq_tokens = seq_tokens[:, 1:, :]
                tokens_per_frame.append(seq_tokens)
        finally:
            if handle is not None:
                handle.remove()
            self._prepool_tokens_tmp = None

        # Stack across time -> (B, T, L, C)
        tokens = torch.stack(tokens_per_frame, dim=1)  # (B,T,L,C)

        if return_grid:
            # Convert patch tokens to spatial grid (B,T,C,H,W)
            if not only_patches and tokens.size(2) > 1:
                tokens = tokens[:, :, 1:, :]
            L = tokens.size(2)
            gh = int(math.sqrt(L))
            if gh * gh != L:
                raise ValueError(f"Cannot reshape {L} tokens into a square grid. Set return_grid=False or only_patches=False.")
            gw = gh
            # (B,T,L,C) -> (B,T,gh,gw,C) -> (B,T,C,gh,gw)
            tokens = tokens.view(B, T, gh, gw, -1).permute(0, 1, 4, 2, 3).contiguous()
            return tokens

        return tokens

def forward_cat_test(output_dim):
    print(f"Concatenation test with output_dim={output_dim}")

    model_name = 'hf-hub:imageomics/bioclip'

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    num_frames = 5
    model = BioCLIPVideoWrapper(
        model_name, output_dim, forward_strat="cat", sequence_length=None, num_frames=num_frames, dropout_rate=0.0
    )
    model.to(device)

    video = torch.randn(8, num_frames, 3, 224, 224).to(device)

    if output_dim is not None:
        target = torch.randn(8, output_dim).to(device)
    else:
        target = torch.randn(8, model.model_output_dim * model.num_frames).to(device)
    criterion = nn.MSELoss()

    output = model(video)
    print(f"output.size(): {output.size()}")

    loss = criterion(output, target)
    print(f"Loss: {loss.item()}")

    loss.backward()

    for name, param in model.named_parameters():
        if param.grad is not None:
            print(f"Gradient for {name}: {param.grad.norm().item()}")
        else:
            print(f"No gradient computed for {name}")

def forward_avg_test(output_dim):
    print(f"Average test with output_dim={output_dim}")

    model_name = 'hf-hub:imageomics/bioclip'

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    num_frames = 5
    model = BioCLIPVideoWrapper(
        model_name, output_dim, forward_strat="average", sequence_length=None, num_frames=num_frames, dropout_rate=0.0
    )
    model.to(device)

    video = torch.randn(8, num_frames, 3, 224, 224).to(device)

    if output_dim is not None:
        target = torch.randn(8, output_dim).to(device)
    else:
        target = torch.randn(8, model.model_output_dim).to(device)
    criterion = nn.MSELoss()

    output = model(video)
    print(f"output.size(): {output.size()}")

    loss = criterion(output, target)
    print(f"Loss: {loss.item()}")

    loss.backward()

    for name, param in model.named_parameters():
        if param.grad is not None:
            print(f"Gradient for {name}: {param.grad.norm().item()}")
        else:
            print(f"No gradient computed for {name}")

def forward_max_test(output_dim):
    print(f"Maximum test with output_dim={output_dim}")

    model_name = 'hf-hub:imageomics/bioclip'

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    num_frames = 1
    model = BioCLIPVideoWrapper(
        model_name, output_dim, forward_strat="max", sequence_length=None, num_frames=num_frames, dropout_rate=0.0
    )
    model.to(device)

    video = torch.randn(8, num_frames, 3, 224, 224).to(device)

    if output_dim is not None:
        target = torch.randn(8, output_dim).to(device)
    else:
        target = torch.randn(8, model.model_output_dim).to(device)
    criterion = nn.MSELoss()

    output = model(video)
    print(f"output.size(): {output.size()}")

    loss = criterion(output, target)
    print(f"Loss: {loss.item()}")

    loss.backward()

    for name, param in model.named_parameters():
        if param.grad is not None:
            print(f"Gradient for {name}: {param.grad.norm().item()}")
        else:
            print(f"No gradient computed for {name}")

def cls_test(output_dim=None):
    print(f"CLS test with output_dim={output_dim}")

    model_name = 'hf-hub:imageomics/bioclip'

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    num_frames = 1
    model = BioCLIPVideoWrapper(
        model_name, output_dim=output_dim, forward_strat="cls", sequence_length=None, num_frames=num_frames, dropout_rate=0.0
    )
    model.to(device)

    video = torch.randn(8, num_frames, 3, 224, 224).to(device)

    if output_dim is not None:
        target = torch.randn(8, output_dim).to(device)
    else:
        target = torch.randn(8, model.model_output_dim).to(device)
    criterion = nn.MSELoss()

    output = model(video)
    print(f"output.size(): {output.size()}")

    loss = criterion(output, target)
    print(f"Loss: {loss.item()}")

    loss.backward()

    for name, param in model.named_parameters():
        if param.grad is not None:
            print(f"Gradient for {name}: {param.grad.norm().item()}")
        else:
            print(f"No gradient computed for {name}")

def print_model_architecture():
    model_name = 'hf-hub:imageomics/bioclip'
    model = BioCLIPVideoWrapper(
        model_name, output_dim=384, forward_strat="cls", 
        sequence_length=None, num_frames=1, dropout_rate=0.0
    )
    print(model)

    for name, param in model.named_parameters():
        param.requires_grad = False

    # Unfreeze last two layers of visual transformer
    for block in model.model.visual.transformer.resblocks[-2:]:
        for param in block.parameters():
            param.requires_grad = True

    # Unfreeze final layers (linear, dropout)
    for param in model.linear.parameters():
        param.requires_grad = True

    # Unfreeze ln_post layer
    model.model.visual.ln_post.weight.requires_grad = True
    model.model.visual.ln_post.bias.requires_grad = True

    # print out .requires_grad for each parameter
    for name, param in model.named_parameters():
        print(f"{name}: {param.requires_grad}")


# --- New: test pre-pooling tokens ---
def test_prepool_tokens():
    print("Testing BioCLIP pre-pooling token extraction...")
    model_name = 'hf-hub:imageomics/bioclip'
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    B, T = 2, 10
    video = torch.randn(B, T, 3, 224, 224).to(device)

    # 1) Use utility method
    model = BioCLIPVideoWrapper(model_name, output_dim=None, num_frames=T).to(device)
    tokens = model.extract_prepool_tokens(video, return_grid=False, only_patches=False)
    print(f"tokens (B,T,L,C): {tuple(tokens.size())}")

    # 2) Grid with only patch tokens
    try:
        grid = model.extract_prepool_tokens(video, return_grid=True, only_patches=True)
        print(f"grid (B,T,C,H,W): {tuple(grid.size())}")
    except Exception as e:
        print(f"grid reshape not possible: {e}")

    # 3) Via forward opt-in
    model_pre = BioCLIPVideoWrapper(model_name, output_dim=None, num_frames=T, return_prepool=True).to(device)
    tokens2 = model_pre(video)
    print(f"tokens via forward (B,T,L,C) or (B,T,C,H,W): {tuple(tokens2.size())}")


if __name__ == "__main__":
    #forward_cat_test(output_dim=None)
    #forward_avg_test(output_dim=None)
    #forward_max_test(output_dim=None)
    #forward_cat_test(output_dim=50)
    #forward_avg_test(output_dim=50)
    #forward_max_test(output_dim=50)

    #cls_test(output_dim=None)
    #cls_test(output_dim=50)

    test_prepool_tokens()

    #print_model_architecture()