import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

class GrooveDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        """
        Args:
            root_dir (string): Directory with all the .npy files.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.root_dir = root_dir
        self.transform = transform
        self.samples = []  # list of file paths

        # Walk through the directory and collect file paths
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

        # Optionally, parse filename metadata if that is useful
        # For example, filename might be "idx_genre_bpm_type_tsnumerator-tsdenominator_bar_X.npy"
        base_filename = os.path.basename(file_path)
        metadata = self.parse_metadata_from_filename(base_filename)

        sample = {"data": tensor_data, "metadata": metadata}
        if self.transform:
            sample = self.transform(sample)

        return sample

    def parse_metadata_from_filename(self, filename):
        # Remove extension and split on underscores.
        # For instance:  "123_rock_120_drum_4-4_bar_1.npy"
        filename = filename.replace(".npy", "")
        parts = filename.split("_")
        metadata = {}
        if len(parts) >= 5:
            metadata["idx"] = parts[0]
            metadata["genre"] = parts[1]
            metadata["bpm"] = parts[2]
            metadata["type"] = parts[3]
            time_sig = parts[4]
            if '-' in time_sig:
                num, denom = time_sig.split("-")
                metadata["time_signature"] = (num, denom)
            # you may parse further as needed.
        return metadata


def main():
    # Specify the directory where your bar arrays are stored.
    dataset_directory = "bar_arrays"
    dataset = GrooveDataset(root_dir=dataset_directory)

    # Create the DataLoader
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True, num_workers=4)

    # Iterate over the DataLoader in your training loop
    for batch in dataloader:
        inputs = batch["data"]           # shape: (batch_size, num_families, num_time_locations)
        metadata = batch["metadata"]     # metadata list for each sample in the batch
        # now perform forward, calculate loss, etc.
        print("Processing batch with input shape:", inputs.shape)


# In your training script or main
if __name__ == '__main__':
    main()
