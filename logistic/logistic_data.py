import json
import numpy as np
from sklearn.model_selection import train_test_split

def get_stats(team: dict[str, int]) -> list:
    """Convert the dictionary representing the statistics of a team for some time in a game into a list."""
    return [team["total_damage_done"],
            team["gold_earned"],
            team["minions_killed"],
            team["jungle_minions_killed"],
            team["xp_gained"],
            team["total_kills"],
            team["tower_kills"],
            team["inhibitor_kills"],
            team["wards_placed"],
            team["wards_killed"],
            team["grubs_killed"],
            team["herald_kills"],
            team["dragon_kills"],
            team["baron_kills"]]


def load_json(filename: str) -> tuple[np.array, np.array]:
    """Load the given game data json into a numpy array.

    The data is given as a list of dicts, each with 6 entries:
        1. Winning team
        2. First blood
        3. Game statistics given per minute as a list of dicts with 3 entries:
            a) The time of the game in minutes
            b) Blue team statistics (14 features)
            c) Red team statistics (14 features)
        4. Blue team champions as a list[int] with 5 elements.
        5. Red team champions as a list[int] with 5 elements.
        6. Match id (ignored for data analysis).

    Returns a numpy array of dimensions N x 41, where N is the sum of the lengths of the games in minutes.
    """
    with open(filename) as f:
        raw_data = json.load(f)
    dataset = []
    labels = []

    for game in raw_data:
        # 1 for blue and 0 for red
        first_blood = 1 if game["first_blood"] == "blue" else 0
        champs = game["blue_team_champions"] + game["red_team_champions"]

        for observation in game["stats_per_minute"]:
            obs_list = [observation["minute"]] + get_stats(observation["blue"]) + get_stats(observation["red"])
            dataset.append([first_blood] + obs_list + champs)
            labels.append(1 if game["winning_team"] == "blue" else 0)

    return np.array(dataset, dtype=np.int64), np.array(labels, dtype=np.int64)


def split_data(dataset: np.array, labels: np.array):
    """Split the dataset and labels into training, validation, and testing datasets."""
    np.random.shuffle(dataset)
    train_data, test_val_data, train_labels, test_val_labels = train_test_split(dataset,
                                                                                labels,
                                                                                test_size=0.3,
                                                                                shuffle=True)
    test_data, val_data, test_labels, val_labels = train_test_split(test_val_data,
                                                                    test_val_labels,
                                                                    test_size=0.5)

    return train_data, val_data, test_data, train_labels, val_labels, test_labels
