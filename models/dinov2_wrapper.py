import torch
import torch.nn as nn

import transformers 
from transformers import AutoModel

class DINOv2VideoWrapper(nn.Module):
    def __init__(self, dino_model_name, output_dim, forward_strat: str="cat", sequence_length=None, num_frames: int=1, dropout_rate=0.1):
        super(DINOv2VideoWrapper, self).__init__()
        
        # Load the DINOv2 model
        self.dino = AutoModel.from_pretrained(dino_model_name)

        self.forward_strat = forward_strat
        # raise value error if forward_strat is not one of the three options.
        if self.forward_strat not in ["cat", "average", "avg", "mean", "max", "maximum"]:
            raise ValueError(f"Invalid forward strategy: {self.forward_strat}. Please use one of 'cat', 'average', or 'max'.")
        self.sequence_length = sequence_length
        self.num_frames = num_frames
        image = torch.randn(1, 3, 224, 224) # (b, c, h, w)
        self._compute_sequence_length(image)
        if self.sequence_length is None and forward_strat == "cat":
            self.sequence_length = self._compute_sequence_length(image)
            self.sequence_length = self.sequence_length*self.num_frames
        
        if self.forward_strat == "cat":
            assert self.sequence_length is not None, f"Sequence length must be provided when using the 'cat' forward strategy. Got {self.sequence_length}"

        # Get the dimension of the output features from DINOv2
        self.dino_output_dim = self.dino.config.hidden_size

        if output_dim is not None:
            # Add a linear layer on top of DINOv2
            self.linear = nn.Linear(
                self.dino_output_dim*self.sequence_length if self.sequence_length is not None else self.dino_output_dim, output_dim
            )
        else:
            self.linear = None
        

        self.dropout1 = nn.Dropout(dropout_rate)
        self.dropout2 = nn.Dropout(dropout_rate)

    def _compute_sequence_length(self, image):
        size_ = self.dino(image[:,:,:,:]).last_hidden_state.size()
        assert len(size_) == 3, f"The output of the DINOv2 model is not of the expected shape. Expected 3 dimensions, got {len(size_)}"
        return size_[1]

    def update_num_frames(self, num_frames):
        self.num_frames = num_frames
        image = torch.randn(1, 3, 224, 224).to(self.dino.device) # (b, c, h, w)
        self.sequence_length = self._compute_sequence_length(image)
        self.sequence_length = self.sequence_length*self.num_frames


    def forward(self, video):

        # video is a list|tensor of images. process each separately 
        # video.size() = batch, #frames, #channels, height, width
        assert len(video.size()) == 5, f"video.size(): {video.size()}; expected 5 dimensions (batch, #frames, #channels, height, width)."
        
        cls_outputs = [self.dino(video[:,i,:,:,:]).last_hidden_state[:, :, :] for i in range(video.size(1))]
        num_frames = len(cls_outputs)

        #batch_size, num_frames, channels, height, width = video.size()
        # Reshape the video tensor to combine the batch and frame dimensions (so can run through everything at once; efficiency improvement)
        # this makes it memory inefficient. 
        #video = video.view(-1, channels, height, width)  # Shape: (batch_size * num_frames, channels, height, width)
        # Pass the entire batch through the model in one go
        #outputs = self.dino(video).last_hidden_state  # Assuming self.dino can handle batch processing
        # Reshape the output to separate the batch and frame dimensions again
        #cls_outputs = outputs.view(batch_size, num_frames, -1, outputs.size(-1))


        assert len(cls_outputs[0].size()) == 3, f"The output of the DINOv2 model is not of the expected shape. Expected 3 dimensions, got {len(cls_outputs[0].size())}"
        
        if self.forward_strat == "cat": 
            output_tensor = torch.cat(cls_outputs, dim=1) # (b, sl * #frames, dm)
            #output_tensor = cls_outputs.view(batch_size, -1, outputs.size(-1))
            
            # flatten.

            # note: sequence length should already count the number of frames.
            output_tensor = output_tensor.view(batch_size, -1) #self.sequence_length*self.dino_output_dim)
            # b, sl*#frames*dm
            output_tensor = self.dropout1(output_tensor) 
        elif self.forward_strat == "average" or self.forward_strat == "avg" or self.forward_strat == "mean":
            stacked_tensors = torch.stack(cls_outputs, dim=1) # (b, #frames, sl, dm)
            #stacked_tensors = self.dropout1(cls_outputs)
            output_tensor = torch.mean(stacked_tensors, dim=-2) # average along sequence length dimension (each patch)
            # (b, #frames, dm)
            output_tensor = torch.mean(output_tensor, dim=-2)# average along frame dimension
            # (b, dm)
        elif self.forward_strat == "max" or self.forward_strat == "maximum":
            stacked_tensors = torch.stack(cls_outputs, dim=1) # (b, #frames, sl, dm)
            #stacked_tensors = self.dropout1(cls_outputs)
            output_tensor = torch.max(stacked_tensors, dim=-2).values # max along sequence length dimension (each patch)
            # (b, #frames, dm)
            output_tensor = torch.max(output_tensor, dim=-2).values # max along frame dimension
            # (b, dm) 
        else: # Error    
            raise ValueError(f"Invalid forward strategy: {self.forward_strat}. Please use one of 'cat', 'average', or 'max'.")
        
        output_tensor = self.dropout2(output_tensor)
        if self.linear is not None:
            linear_output = self.linear(output_tensor)
        else:
            linear_output = output_tensor

        return linear_output

def forward_cat_test():

    print(f"Concatentation test")

    dino_model_name = 'facebook/dinov2-base'
    output_dim = 50

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    num_frames = 5
    model = DINOv2VideoWrapper(
        dino_model_name, output_dim, forward_strat="cat", sequence_length=None, num_frames=num_frames, dropout_rate=0.0
    )
    model.to(device)

    video = torch.randn(8, num_frames, 3, 224, 224).to(device)

    # Define a dummy target and loss function
    target = torch.randn(8, output_dim).to(device)  # Assuming output_dim matches the target size
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


def forward_avg_test():
    
    print(f"Average test")

    dino_model_name = 'facebook/dinov2-base'
    output_dim = 50

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    num_frames = 5
    model = DINOv2VideoWrapper(
        dino_model_name, output_dim, forward_strat="average", sequence_length=None, num_frames=num_frames, dropout_rate=0.0
    )
    model.to(device)

    video = torch.randn(8, num_frames, 3, 224, 224).to(device)

    # Define a dummy target and loss function
    target = torch.randn(8, output_dim).to(device)  # Assuming output_dim matches the target size
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

def forward_max_test():
    
    print(f"Maximum test")

    dino_model_name = 'facebook/dinov2-base'
    output_dim = 50

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    num_frames = 5
    model = DINOv2VideoWrapper(
        dino_model_name, output_dim, forward_strat="max", sequence_length=None, num_frames=num_frames, dropout_rate=0.0
    )
    model.to(device)

    video = torch.randn(8, num_frames, 3, 224, 224).to(device)

    # Define a dummy target and loss function
    target = torch.randn(8, output_dim).to(device)  # Assuming output_dim matches the target size
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

if __name__ == "__main__":
    
    #forward_cat_test()
    forward_avg_test()
    #forward_max_test()