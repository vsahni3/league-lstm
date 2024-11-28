Predict probability of winning League of Legends Match at any point of the game

Instructions:
1. Store the json data <file.json> locally
2. Install requirements in venv
3. Call convert_json_to_jsonl('file.json', 'file.jsonl') to convert to jsonl
4. Call precompute_jsonl_offsets("file.jsonl") to store byte offsets for O(1) access, note this should create a pkl file
5. Now you are good to go to run the training loop
