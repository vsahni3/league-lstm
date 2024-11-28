
import json
import pickle
from torch.utils.data import Dataset
import torch
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


class LeagueDataset(Dataset):
    def __init__(self, jsonl_file, offsets_file):
        self.jsonl_file = jsonl_file
        with open(offsets_file, 'rb') as f:
            self.offsets = pickle.load(f)

    def __len__(self):
        """Return the total number of samples."""
        return len(self.offsets)

    def __getitem__(self, idx):
        """Retrieve a sample and its label by index."""
        data = access_jsonl_by_offset(self.jsonl_file, self.offsets, idx)
        label = torch.tensor(data['winning_team'] == 'blue', dtype=torch.float)
        x = [list(entry['blue'].values()) + list(entry['red'].values()) for entry in data['stats_per_minute']]
        x = torch.tensor(x, dtype=torch.float32)
        return x, label


def collate_fn(batch):
    """
    Custom collate function for variable-length sequences.
    Args:
        batch (list of tuples): List of (sample, label) tuples.
    Returns:
        padded_samples: Tensor of padded samples.
        labels: Tensor of labels.
    """
    samples, labels = zip(*batch)

    # Pad sequences to the same length
    padded_samples = torch.nn.utils.rnn.pad_sequence(
        [torch.tensor(sample) for sample in samples],
        batch_first=True,
        padding_value=0
    )
    labels = torch.tensor(labels)
    return padded_samples, labels

# Example Usage

# convert_json_to_jsonl('combined_data.json', 'combined_data.jsonl')
# precompute_jsonl_offsets("combined_data.jsonl")
# ids = set()
# for i in range(57099):
#     element = access_jsonl_by_offset("combined_data.jsonl", "offsets.pkl", i)['match_id']
#     assert element not in ids 
#     ids.add(element)


