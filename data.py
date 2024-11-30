
import json
import pickle
from torch.utils.data import Dataset
import torch
from functools import lru_cache
import bisect

def convert_json_to_jsonl(json_file, jsonl_file):
    """
    Convert a large JSON list to JSONL format.

    Args:
        json_file (str): Path to the input JSON file.
        jsonl_file (str): Path to the output JSONL file.
    """
    with open(json_file, 'r') as f_in, open(jsonl_file, 'w') as f_out:
        data = json.load(f_in)  # Load the JSON list
        for element in data:
            f_out.write(json.dumps(element) + '\n')  # Write each dict as a line

def precompute_jsonl_offsets(jsonl_file):
    """
    Precompute byte offsets for each line in a JSONL file.

    Args:
        jsonl_file (str): Path to the JSONL file.
        offsets_file (str): Path to the file where offsets will be stored.
    """
    offsets = []
    with open(jsonl_file, 'r') as f:
        while True:
            pointer = f.tell()
            line = f.readline()
            if not line:
                break
            offsets.append(pointer)

    with open('offsets.pkl', 'wb') as f:
        pickle.dump(offsets, f)


def access_jsonl_by_offset(jsonl_file, offsets, line_index):
    with open(jsonl_file, 'r') as f:
        f.seek(offsets[line_index])  # Jump to the desired byte offset
        line = f.readline()
        return json.loads(line)  # Parse and return the JSON object

def compute_index_map(jsonl_file, offsets, fraction):
    indices = []
    count = 0
    for i in range(len(offsets)):
        data = access_jsonl_by_offset(jsonl_file, offsets, i)['stats_per_minute']
        start_idx = round(len(data) * fraction)
        count += len(data) - start_idx
        indices.append(count)
    with open('indices.pkl', 'wb') as f:
        pickle.dump(indices, f)

# with open('offsets.pkl', 'rb') as f:
#     offsets = pickle.load(f)
# compute_index_map('combined_data.jsonl', offsets, 0.8)

class LeagueDataset(Dataset):
    def __init__(self, jsonl_file, offsets_file, indices_file):
        self.jsonl_file = jsonl_file
        with open(offsets_file, 'rb') as f:
            self.offsets = pickle.load(f)
        with open(indices_file, 'rb') as f:
            self.indices = pickle.load(f)
        
    def __len__(self):
        """Return the total number of samples."""
        return self.indices[-1] // 10

    def __getitem__(self, idx):
        """Retrieve a sample and its label by index."""
        # use binary search to find json index
        index = bisect.bisect_right(self.indices, idx)
        prev_elements = self.indices[index - 1] if index > 0 else 0
        num_elements = self.indices[index] - prev_elements
        cur_elements = idx - prev_elements
        difference = num_elements - cur_elements - 1
        data = access_jsonl_by_offset(self.jsonl_file, self.offsets, index)
        stats = data['stats_per_minute']
        label = torch.tensor(data['winning_team'] == 'blue', dtype=torch.float)
        x = [
            list(entry['blue'].values()) + list(entry['red'].values()) 
            for entry in stats[:len(stats) - difference]
            ]
        x = torch.tensor(x, dtype=torch.float32)
        return x, x.shape[0], len(stats), label

def collate_fn(batch):
    """
    Custom collate function for variable-length sequences.
    Args:
        batch (list of tuples): List of (sample, label) tuples.
    Returns:
        padded_samples: Tensor of padded samples.
        labels: Tensor of labels.
    """
    samples, lengths, total_lengths, labels = zip(*batch)

    # Pad sequences to the same length
    padded_samples = torch.nn.utils.rnn.pad_sequence(
        [torch.tensor(sample) for sample in samples],
        batch_first=True,
        padding_value=0
    )
    labels = torch.tensor(labels)
    return padded_samples, lengths, total_lengths, labels

# Example Usage

# convert_json_to_jsonl('combined_data.json', 'combined_data.jsonl')
# precompute_jsonl_offsets("combined_data.jsonl")
# ids = set()
# for i in range(57099):
#     element = access_jsonl_by_offset("combined_data.jsonl", "offsets.pkl", i)['match_id']
#     assert element not in ids 
#     ids.add(element)


