import torch
from torch.utils.data import Dataset, DataLoader
import random
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
import h5py
import glob
import os
from collections import defaultdict
from torchvision.transforms.v2.functional import InterpolationMode

class AnimalClipDataset(Dataset):
    def __init__(
        self, directory, animal_cooccurrences,
        transforms=None, K=20, num_frames=10, 
        total_frames=20, mode="positive_negative",  
        zfill_num=4, override_value=None
    ):
        """
        Args:
            directory: path to where to import data.
            animal_cooccurrences (dict): Precomuted dictionary of {animal_id: [cooccurring_animal_ids]}.
            transforms (callable, optional): Optional transform to be applied on a sample.
            K (int): Number of clips to select for positive and negative sets.
            num_frames (int): Number of frames to subsample from dataset. 1 for image dataset. (Default = 10)
            mode (str): Mode of the dataset. Options are "positive_negative" or "video_only". (Default = "positive_negative")
            total_frames (int): Total number of frames in a clip. (Default = 20)
        """
        self.clip_metadata = defaultdict(list)                  #clip_metadata (dict): Dictionary of clip ids by animal_id {animal_id: [clip_ids]}.
        self.clips = dict()                                     #clips (dict): Dictionary holding clip arrays {clip_ids: numpy_array}.
        self.animal_cooccurrences = animal_cooccurrences

        ## clip_paths to a list of clip paths from a dict. Get values into a dict...
        self.transformations = transforms
        self.total_frames = total_frames #For meerkats the dataset is 20 frames total
        self.K = K
        self.num_frames = num_frames
        self.mode = mode
        self.zfill_num = zfill_num
        self.load_all_videos(directory)
        self.list_clip_paths = [k for k in self.clips.keys()]
        self.override_value = override_value
        

    def __len__(self):
        if self.mode == "positive_negative":
            return sum(len(clips) for clips in self.clip_metadata.values())
        elif self.mode == "video_only" or self.mode == "video":
            return len(self.list_clip_paths)
        elif self.mode == "override":
            assert self.override_value is not None and isinstance(self.override_value, int), "Override value must be set to an integer"
            return self.override_value
        else:
            raise Exception(f"Mode ({self.mode}) not recognized")
        
    def load_all_videos(self, directory):
        """
        Read the h5 dataset files into a dictionary of numpy arrays
        """
        # Calculate the interval between frames to be selected
        interval = np.linspace(0, self.total_frames-1, self.num_frames, dtype=int)
        for filepath in glob.glob(os.path.join(directory, "*.h5")):
            with h5py.File(filepath, 'r') as h5_file:
                # Convert the HDF5 file to a dictionary
                if self.num_frames != 1:
                    for key in h5_file.keys():
                        #Sample frames when loading the dataset to save memory
                        self.clips[key] = np.asarray(h5_file[key])[interval,...]
                        self.clip_metadata[str(int(key.split("_")[0])).zfill(self.zfill_num)] += [key]
                else:
                    for key in h5_file.keys():
                        self.clips[key] = np.asarray(h5_file[key])[0,:,:,:][np.newaxis, :, :, :]
                        self.clip_metadata[str(int(key.split("_")[0])).zfill(self.zfill_num)] += [key]
        #print(f"meta_data: {self.clip_metadata.keys()}")
        print(f"Total number of animals: {len(self.clip_metadata.keys())}")
        print(f"Total number of clips: {len(self.clips.keys())}")

    def video_only(self, idx):
        # Implement this method to load a clip given its index in the list of clips.
        path = self.list_clip_paths[idx]
        return path, self.load_clip(path)
    
    def load_clip(self, clip_id):
        # Implement this method to load a clip given its ID.
        video = self.clips[clip_id]
        video = torch.from_numpy(video.astype(np.float32).transpose((0, 3, 1, 2))).contiguous()/255.0
        if self.transformations:
            video = self.transformations(video)
            video = (video-video.amin(keepdim=True)) / (video.amax(keepdim=True)-video.amin(keepdim=True))
        if self.num_frames != 1:
            return video
        return video[0,...]
    
    def positive_negative_clips(self):
        # Randomly select an individual with at least 3 videos
        negative_clips = []
        while len(negative_clips) < 3:
            positive_clips = []
            while len(positive_clips) < 3:
                # Select a random anchor animal
                anchor_animal_id = random.choice(list(self.clip_metadata.keys()))
                positive_clips = self.clip_metadata[anchor_animal_id]

            # Select up to K positive clips (same animal)
            positive_clip_ids = [clip_id for clip_id in positive_clips]
            positive_clip_ids = random.sample(positive_clip_ids, min(len(positive_clip_ids), self.K))

            # Select up to K negative clips (co-occurring animals)
            cooccurring_animal_ids = self.animal_cooccurrences[str(int(anchor_animal_id))]
            cooccurring_clip_ids = []

            for animal_id in cooccurring_animal_ids:
                cooccurring_clip_ids.extend(self.clip_metadata.get(str(animal_id).zfill(self.zfill_num), []))
            negative_clips = random.sample(cooccurring_clip_ids, min(len(cooccurring_clip_ids), self.K))
        
        negative_clips = [self.load_clip(clip_id) for clip_id in negative_clips]
        positive_clips = [self.load_clip(clip_id) for clip_id in positive_clip_ids]

        return positive_clips, negative_clips

    def __getitem__(self, idx):
        if self.mode == "positive_negative":
            return self.positive_negative_clips() 
        elif self.mode == "video_only" or self.mode == "video":
            return self.video_only(idx)
        else:
            raise Exception(f"Mode ({self.mode}) not recognized")
       
if __name__ == "__main__":
    import json
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    from PIL import Image
    import torchvision.transforms.v2 as transforms

    #Read the precomputed cooccurences dictionary
    with open('../Dataset/meerkat_h5files/Cooccurrences.json', 'r') as file:
        cooccurrences = json.load(file)

    def create_gif(frames_folder, gif_path, duration=3):
        frame_files = sorted(os.listdir(frames_folder))  # Get list of frame files sorted alphabetically

        frames = []
        for frame_file in frame_files:
            frame_path = os.path.join(frames_folder, frame_file)
            frame = Image.open(frame_path)
            frames.append(frame)

        # Save as GIF
        frames[0].save(gif_path, save_all=True, append_images=frames[1:], loop=0, duration=duration)
    
    s=1/8.0 #Based on SimCLR strength parameter for supervised learning
    color_jitter = transforms.ColorJitter(0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s)
    gaussian = transforms.GaussianBlur(kernel_size=23) #Based on SimCLR, 10% of the image size, sigma = 0.1, 2.0
    meerkat_transforms = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0), antialias=True, interpolation=InterpolationMode.BICUBIC), #SimCLR, BYOL
        transforms.RandomHorizontalFlip(),                  #50% probability based on SimCLR
        transforms.RandomApply([gaussian], p=0.5),          #50% probability based on SimCLR
        transforms.RandomApply([color_jitter], p=0.8)       #Based on SimCLR but without color drop (grayscale)
    ])
    dataset = AnimalClipDataset("/home/kkno604/data/meerkat_data/h5files/Train/", cooccurrences, num_frames=20, transforms=meerkat_transforms)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True) # TODO test larger batch size

    print("Data loaded")
    data_iterator = iter(dataloader)
    for j in range(20):
        positive_clips, negative_clips = next(data_iterator)
        anchor_clip = positive_clips[0]

        print(anchor_clip.shape, len(positive_clips), len(negative_clips))
        '''
        for k in range(20): #Change this to clip length
            fig, ((ax1),(ax2)) = plt.subplots(1, 2)
            ax1.axis('off')
            ax2.axis('off')

            gs2 = gridspec.GridSpecFromSubplotSpec(5, 4, subplot_spec=ax1)
            axes = []
            for i, img in enumerate(positive_clips):
                ax = fig.add_subplot(gs2[i // 4, i % 4])
                im = img[0,k,:,:,:].cpu().numpy()
                im = im.transpose(1,2,0)
                ax.imshow(im)
                ax.axis('off')
                axes += [ax]
            ax1.set_title("Positive observations")
            
            plt.subplots_adjust(wspace=0.1, hspace=0.1)

            gs3 = gridspec.GridSpecFromSubplotSpec(5, 4, subplot_spec=ax2)
            n_imgs_section3 = len(negative_clips)
            axes = []
            for i, img in enumerate(negative_clips):
                ax = fig.add_subplot(gs3[i // 4, i % 4])
                im = img[0,k,:,:,:].cpu().numpy()
                im = im.transpose(1,2,0)
                ax.imshow(im)
                ax.axis('off')
                axes += [ax]
            ax2.set_title("Negative observations")

            plt.subplots_adjust(wspace=0.1, hspace=0.1)

            #plt.show()
            plt.savefig("Gif/"+str(k).zfill(2)+".png")
            plt.close()
        create_gif("Gif/", "Gif"+str(j).zfill(3)+str(i).zfill(3)+".gif")
        '''