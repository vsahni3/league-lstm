import ml_collections

def get_league_config():
    config = ml_collections.ConfigDict()
    config.input_size = 28
    config.hidden_size = 64
    return config