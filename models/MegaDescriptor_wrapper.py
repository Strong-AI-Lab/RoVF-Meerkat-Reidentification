import torch
import torch.nn as nn
import timm
import os

class MegaDescriptorVideoWrapper(nn.Module):
    def __init__(
        self, model_name, output_dim, forward_strat: str="cat", sequence_length=None, 
        num_frames: int=1, dropout_rate=0.1, pretrained=True, checkpoint_path=None
    ):
        super(MegaDescriptorVideoWrapper, self).__init__()
        
        # Load the MegaDescriptor model
        self.model = timm.create_model(model_name, pretrained=pretrained)
        if checkpoint_path:
            self.model.load_state_dict(torch.load(checkpoint_path))

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

        # Get the dimension of the output features from MegaDescriptor
        self.model_output_dim = self.model.num_features

        if output_dim is not None:
            #print(f"Assuming cls or cat strategy for linear layer at the end.")
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
        size_ = self.model(image).size()
        assert len(size_) == 2, f"The output of the MegaDescriptor model is not of the expected shape. Expected 2 dimensions, got {len(size_)}"
        return size_[1]

    def forward(self, video):
        if len(video.size()) == 4: # assume that this is because num_frames=1
            video = video.unsqueeze(1)
        assert len(video.size()) == 5, f"video.size(): {video.size()}; expected 5 dimensions (batch, #frames, #channels, height, width)."
        
        #print(f"self.model(video[:,i,:,:,:]): {self.model(video[:,0,:,:,:]).size()}")

        cls_outputs = [self.model(video[:,i,:,:,:]) for i in range(video.size(1))] # [#frames] [b, dm]
        num_frames = len(cls_outputs)

        batch_size = video.size(0)

        assert len(cls_outputs[0].size()) == 2, f"The output of the MegaDescriptor model is not of the expected shape. Expected 2 dimensions, got {len(cls_outputs[0].size())}"
        
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
            #output_tensor = torch.stack([cls_[:,0] for cls_ in cls_outputs], dim=1) # (b, #frames, dm)
            #output_tensor = torch.mean(output_tensor, dim=1)
            output_tensor = cls_outputs[-1] # (b, dm) # if this is a video, take the last frame. Assume in practice that only one frame is provided.
        else:    
            raise ValueError(f"Invalid forward strategy: {self.forward_strat}. Please use one of 'cat', 'average', or 'max'.")
        
        output_tensor = self.dropout2(output_tensor)
        if self.linear is not None:
            linear_output = self.linear(output_tensor)
        else:
            linear_output = output_tensor

        return linear_output

def forward_cat_test(output_dim):

    print(f"Concatenation test with output_dim={output_dim}")

    model_name = 'hf-hub:BVRA/MegaDescriptor-T-224'

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    num_frames = 5
    model = MegaDescriptorVideoWrapper(
        model_name, output_dim, forward_strat="cat", sequence_length=None, num_frames=num_frames, dropout_rate=0.0
    )
    model.to(device)

    video = torch.randn(8, num_frames, 3, 224, 224).to(device)

    # Define a dummy target and loss function
    if output_dim is not None:
        target = torch.randn(8, output_dim).to(device)  # Assuming output_dim matches the target size
    else:
        target = torch.randn(8, model.model_output_dim * model.num_frames).to(device)  # Use model_output_dim * sequence_length if output_dim is None
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

    # Save checkpoint
    checkpoint_path = "megadescriptor_checkpoint.pth"
    torch.save(model.state_dict(), checkpoint_path)
    print(f"Checkpoint saved to {checkpoint_path}")

    # Load checkpoint
    model.load_state_dict(torch.load(checkpoint_path))
    print(f"Checkpoint loaded from {checkpoint_path}")

    # delete checkpoint
    os.remove(checkpoint_path)


def forward_avg_test(output_dim):
    
    print(f"Average test with output_dim={output_dim}")

    model_name = 'hf-hub:BVRA/MegaDescriptor-T-224'

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    num_frames = 5
    model = MegaDescriptorVideoWrapper(
        model_name, output_dim, forward_strat="average", sequence_length=None, num_frames=num_frames, dropout_rate=0.0
    )
    model.to(device)

    video = torch.randn(8, num_frames, 3, 224, 224).to(device)

    # Define a dummy target and loss function
    if output_dim is not None:
        target = torch.randn(8, output_dim).to(device)  # Assuming output_dim matches the target size
    else:
        target = torch.randn(8, model.model_output_dim).to(device)  # Use model_output_dim if output_dim is None
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

def forward_max_test(output_dim):
    
    print(f"Maximum test with output_dim={output_dim}")

    model_name = 'hf-hub:BVRA/MegaDescriptor-T-224'

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    num_frames = 5
    model = MegaDescriptorVideoWrapper(
        model_name, output_dim, forward_strat="max", sequence_length=None, num_frames=num_frames, dropout_rate=0.0
    )
    model.to(device)

    video = torch.randn(8, num_frames, 3, 224, 224).to(device)

    # Define a dummy target and loss function
    if output_dim is not None:
        target = torch.randn(8, output_dim).to(device)  # Assuming output_dim matches the target size
    else:
        target = torch.randn(8, model.model_output_dim).to(device)  # Use model_output_dim if output_dim is None
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

def test_cls(output_dim):
    print(f"CLS test with output_dim={output_dim}")

    model_name = 'hf-hub:BVRA/MegaDescriptor-T-224'

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    num_frames = 5
    model = MegaDescriptorVideoWrapper(
        model_name, output_dim, forward_strat="cls", sequence_length=None, num_frames=num_frames, dropout_rate=0.0
    )
    model.to(device)

    video = torch.randn(8, num_frames, 3, 224, 224).to(device)

    # Define a dummy target and loss function
    if output_dim is not None:
        target = torch.randn(8, output_dim).to(device)  # Assuming output_dim matches the target size
    else:
        target = torch.randn(8, model.model_output_dim).to(device)  # Use model_output_dim if output_dim is None
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
    model_name = 'hf-hub:BVRA/MegaDescriptor-L-224'
    model = MegaDescriptorVideoWrapper(
        model_name, output_dim=384, forward_strat="cls", 
        sequence_length=None, num_frames=1, dropout_rate=0.0
    )
    print(model)

    for name, param in model.named_parameters():
        param.requires_grad = False

    # Freeze all layers except for the last SwinTransformerStage (model.layers.3), norm, head, and linear layers
    for name, param in model.named_parameters():
        if (name.startswith("model.layers.3") or 
                name.startswith("model.norm") or 
                name.startswith("model.head") or 
                name.startswith("linear")):
            param.requires_grad = True

    # Verify which layers are trainable
    for name, param in model.named_parameters():
        print(f"{name}: requires_grad={param.requires_grad}")


if __name__ == "__main__":
    
    #forward_cat_test(output_dim=None)
    #forward_avg_test(output_dim=None)
    #forward_max_test(output_dim=None)
    #forward_cat_test(output_dim=50)
    #forward_avg_test(output_dim=50)
    #forward_max_test(output_dim=50)

    #test_cls(1000)
    #test_cls(None)

    print_model_architecture()