import torch
import torch.nn as nn
import open_clip

class BioCLIPVideoWrapper(nn.Module):
    def __init__(self, model_name, output_dim, forward_strat: str="cat", sequence_length=None, num_frames: int=1, dropout_rate=0.1, checkpoint_path=None):
        super(BioCLIPVideoWrapper, self).__init__()
        
        # Load the BioCLIP model
        self.model, self.preprocess_train, self.preprocess_val = open_clip.create_model_and_transforms(model_name)
        self.tokenizer = open_clip.get_tokenizer(model_name)
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

    num_frames = 5
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
    model = BioCLIPVideoWrapper(model_name, output_dim=None, forward_strat="cls", sequence_length=None, num_frames=5, dropout_rate=0.0)
    print(model)

if __name__ == "__main__":
    #forward_cat_test(output_dim=None)
    #forward_avg_test(output_dim=None)
    #forward_max_test(output_dim=None)
    #forward_cat_test(output_dim=50)
    #forward_avg_test(output_dim=50)
    forward_max_test(output_dim=50)

    cls_test(output_dim=None)
    #cls_test(output_dim=50)

    print_model_architecture()