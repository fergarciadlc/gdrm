import os
import json
import numpy as np
import torch
import torch.nn.functional as f
from torch.utils.data import Dataset, DataLoader

class GrooveDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        """
        Args:
            root_dir (string): Directory with all the .npy files and the genre_mapping.json.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.root_dir = root_dir
        self.transform = transform
        self.samples = []  # list of file paths

        # Load the genre mapping from genre_mapping.json
        mapping_path = os.path.join(root_dir, "genre_mapping.json")
        if os.path.exists(mapping_path):
            with open(mapping_path, "r") as f:
                self.genre_mapping = json.load(f)
        else:
            self.genre_mapping = {}  # fallback empty mapping if file not found

        # Walk through the directory and collect file paths for .npy files
        for dirpath, _, filenames in os.walk(root_dir):
            for file in filenames:
                if file.endswith(".npy"):
                    full_path = os.path.join(dirpath, file)
                    self.samples.append(full_path)

        # Optionally sort to ensure a consistent order
        self.samples.sort()

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        file_path = self.samples[idx]
        # Load numpy array (which was created by save_bar_arrays)
        data = np.load(file_path)

        # Convert to a torch tensor (if needed, cast type)
        tensor_data = torch.from_numpy(data).float()  # convert to float if needed

        # Parse filename metadata. For example, filename might be "123_rock-jazz_120_drum_4-4_bar_1.npy"
        base_filename = os.path.basename(file_path)
        metadata = self.parse_metadata_from_filename(base_filename)

        sample = {"data": tensor_data, "metadata": metadata}
        if self.transform:
            sample = self.transform(sample)

        return sample

    def parse_metadata_from_filename(self, filename):
        # Remove extension and split on underscores.
        # For instance: "123_rock-jazz_120_drum_4-4_bar_1.npy"
        filename = filename.replace(".npy", "")
        parts = filename.split("_")
        metadata = {}
        if len(parts) >= 5:
            metadata["idx"] = parts[0]
            # Get the genre and remove subgenre information if present.
            genre_str = parts[1]
            # If the genre has a hyphen, keep only the part before the hyphen.
            if '-' in genre_str:
                genre_str = genre_str.split('-')[0]
            # Fetch the genre index from the mapping (defaulting to 0 if not found).
            genre_idx = self.genre_mapping.get(genre_str, 0)
            # Ensure the genre index is converted to a tensor with the correct type for one_hot encoding.
            metadata["genre"] = f.one_hot(torch.tensor(genre_idx, dtype=torch.long), num_classes=len(self.genre_mapping))
            bpm_int = int(parts[2])
            normalised_bpm = 2 * ((bpm_int - 60) / (200 - 60)) - 1
            metadata["bpm"] = torch.tensor(normalised_bpm)
            metadata["type"] = parts[3]
            time_sig = parts[4]
            if '-' in time_sig:
                num, denom = time_sig.split("-")
                metadata["time_signature"] = (num, denom)
        return metadata


def main():
    # Specify the directory where your bar arrays and the genre mapping file are stored.
    dataset_directory = "bar_arrays"
    dataset = GrooveDataset(root_dir=dataset_directory)

    # Create the DataLoader
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True, num_workers=4)

    # Iterate over the DataLoader in your training loop
    for batch in dataloader:
        inputs = batch["data"]           # shape: (batch_size, num_families, num_time_locations)
        genre_data = batch["metadata"]["genre"]         # shape: (batch, 18) one-hot
        bpm_data = batch["metadata"]["bpm"].unsqueeze(dim=1)             # shape: (batch, 1); ensure it is normalized

        # now perform forward, calculate loss, etc.
        print("Processing batch with input shape:", inputs.shape)
        print("genre_data shape: ", genre_data.shape)
        print("bpm_data shape: ", bpm_data.shape)

# In your training script or main
if __name__ == '__main__':
    main()
