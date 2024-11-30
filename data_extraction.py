import requests
import json
import os
import time
import sqlite3
import threading
import math
from tqdm import tqdm

API_KEYS = [


]
DIVISIONS = [
    "PLATINUM/I",
    "PLATINUM/II"
]


class RiotAPI:
    def __init__(self, api_keys):
        self.api_keys = api_keys
        self.current_key_index = 0
        self.first_use_time = None

    def get_current_api_key(self):
        return self.api_keys[self.current_key_index]

    def rotate_api_key(self):
        self.current_key_index = (self.current_key_index + 1) % len(self.api_keys)
        print(f"switched api key to {self.get_current_api_key()}")

        # check if we've cycled back to the first key
        if self.current_key_index == 0:
            # calculate the elapsed time since the first use in this cycle
            elapsed_time = time.time() - self.first_use_time
            wait_time = max(0, 125 - elapsed_time)  # wait the remainder to reach 125 seconds

            if wait_time > 0:
                print(f"cycled through all api keys. waiting {wait_time:.2f} seconds before continuing.")
                time.sleep(wait_time)

            # reset the first use time after completing a cycle
            self.first_use_time = None

    def make_request(self, url):
        if self.first_use_time is None:
            # record the first use time if not already set
            self.first_use_time = time.time()

        while True:
            api_key = self.get_current_api_key()
            headers = {"X-Riot-Token": api_key}
            response = requests.get(url, headers=headers)

            if response.status_code == 200:
                return response.json()
            elif response.status_code == 429:
                print("rate limit reached even with preemptive rotating.")
                time.sleep(120)
            else:
                response.raise_for_status()

def get_summoner_ids(division, page, offset=0, num_ids=25):
    url = f"https://na1.api.riotgames.com/lol/league/v4/entries/RANKED_SOLO_5x5/{division}?page={page}"
    data = riot_api.make_request(url)
    summoner_ids = [entry["summonerId"] for entry in data[offset:offset + num_ids]]
    return summoner_ids

def get_puuid_by_summoner_id(summoner_id):
    url = f"https://na1.api.riotgames.com/lol/summoner/v4/summoners/{summoner_id}"
    data = riot_api.make_request(url)
    return data.get("puuid")

def get_top_matches_by_puuid(puuid, count=10):
    url = f"https://americas.api.riotgames.com/lol/match/v5/matches/by-puuid/{puuid}/ids?queue=420&type=ranked&start=0&count={count}"
    return riot_api.make_request(url)

def save_match_ids(match_ids, filename="match_ids.json"):
    directory = "data"
    os.makedirs(directory, exist_ok=True)
    filepath = os.path.join(directory, filename)

    if os.path.exists(filepath):
        with open(filepath, "r") as file:
            existing_ids = set(json.load(file))
    else:
        existing_ids = set()

    all_match_ids = existing_ids.union(match_ids)
    with open(filepath, "w") as file:
        json.dump(list(all_match_ids), file, indent=4)
    print(f"saved matches, current amount: {len(all_match_ids)}")

def process_divisions(max_pages=20, ids_per_page=200, ids_per_batch=25):
    match_ids_to_save = set()

    for division in DIVISIONS:

        for page in range(1, max_pages + 1):
            print(f"\n\n\n =========processing division {division}, page {page}=========")

            # track ids retrieved per page
            ids_retrieved = 0
            offset = 0

            for _ in range(ids_per_page // ids_per_batch):
                summoner_ids = get_summoner_ids(division, page, offset=offset, num_ids=ids_per_batch)
                print(f"found {len(summoner_ids)} summoner ids")
                if not summoner_ids:
                    break  # no more summoner ids on this page

                for summoner_id in tqdm(summoner_ids, desc="processing summoners", unit="summoner"):
                    puuid = get_puuid_by_summoner_id(summoner_id)
                    if puuid:
                        match_ids = get_top_matches_by_puuid(puuid, count=10)
                        match_ids_to_save.update(match_ids)

                ids_retrieved += len(summoner_ids)
                offset += 25  # move to the next set of 25 ids on the page

                print(f"retrieved {ids_retrieved}/{ids_per_page} summoner ids for division {division}, page {page}")

                # save match ids
                save_match_ids(match_ids_to_save, filename="match_ids.json")

                riot_api.rotate_api_key()  # rotate api key after match id retrieval

def get_match_timeline(match_id, riot_api: RiotAPI):
    base_url = f"https://americas.api.riotgames.com/lol/match/v5/matches/{match_id}/timeline"
    return riot_api.make_request(base_url)


def process_match_timeline(match_id, riot_api: RiotAPI):
    # retrieve match timeline data
    timeline_data = get_match_timeline(match_id, riot_api)

    if not timeline_data:
        return None  # if timeline data is not retrieved, return none

    # confirm the game completion status
    if timeline_data["info"].get("endOfGameResult") != "GameComplete":
        return None

    frames = timeline_data["info"]["frames"]
    last_frame = frames[-1]
    match_duration = last_frame["timestamp"]  # duration in milliseconds

    if match_duration < 900000:  # 15 minutes in milliseconds
        return None

    # find the winning team
    winning_team = None
    for event in last_frame.get("events", []):
        if event.get("type") == "GAME_END":
            winning_team = "blue" if event["winningTeam"] == 100 else "red"

    # check for first blood
    first_blood = None  # either "blue" or "red"
    for frame in frames:
        for event in frame.get("events", []):
            if event.get("type") == "CHAMPION_SPECIAL_KILL" and event.get("killType") == "KILL_FIRST_BLOOD":
                killer_id = event.get("killerId")
                if killer_id:
                    first_blood = "blue" if killer_id <= 5 else "red"
                break
        if first_blood:  # stop searching once first blood is found
            break
    if first_blood is None:
        return None

    # initialize a list to store stats per minute
    stats_per_minute = []

    # process stats for each minute of the match
    for minute in range(0, match_duration // 60000 + 2):  # convert milliseconds to minutes
        aggregated_stats = {
            "minute": minute,
            "blue": {
                "total_damage_done": 0,
                "gold_earned": 0,
                "minions_killed": 0,
                "jungle_minions_killed": 0,
                "xp_gained": 0,
                "total_kills": 0,
                "tower_kills": 0,
                "inhibitor_kills": 0,
                "wards_placed": 0,
                "wards_killed": 0,
                "grubs_killed": 0,
                "herald_kills": 0,
                "dragon_kills": 0,
                "baron_kills": 0
            },
            "red": {
                "total_damage_done": 0,
                "gold_earned": 0,
                "minions_killed": 0,
                "jungle_minions_killed": 0,
                "xp_gained": 0,
                "total_kills": 0,
                "tower_kills": 0,
                "inhibitor_kills": 0,
                "wards_placed": 0,
                "wards_killed": 0,
                "grubs_killed": 0,
                "herald_kills": 0,
                "dragon_kills": 0,
                "baron_kills": 0
            }
        }

        # get the most recent frame for the current minute
        current_frame = None
        for frame in frames:
            if frame["timestamp"] > minute * 60000:
                break
            current_frame = frame

        # if a valid frame exists, aggregate participant frame stats
        if current_frame:
            for participant_id, participant_data in current_frame["participantFrames"].items():
                team = "blue" if int(participant_id) <= 5 else "red"

                aggregated_stats[team]["total_damage_done"] += participant_data["damageStats"][
                    "totalDamageDoneToChampions"]
                aggregated_stats[team]["gold_earned"] += participant_data["totalGold"]
                aggregated_stats[team]["minions_killed"] += participant_data["minionsKilled"]
                aggregated_stats[team]["jungle_minions_killed"] += participant_data["jungleMinionsKilled"]
                aggregated_stats[team]["xp_gained"] += participant_data["xp"]

        # loop through all frames up to the current minute
        for frame in frames:
            if frame["timestamp"] > minute * 60000:
                break

            # check events in the current frame
            for event in frame.get("events", []):
                if event.get("type") == "CHAMPION_KILL":
                    killer_id = event.get("killerId")
                    if killer_id:
                        if killer_id <= 5:
                            aggregated_stats["blue"]["total_kills"] += 1
                        else:
                            aggregated_stats["red"]["total_kills"] += 1
                elif event.get("type") == "ELITE_MONSTER_KILL":
                    if event.get("monsterType") == "HORDE":
                        aggregated_stats["blue" if event["killerTeamId"] == 100 else "red"]["grubs_killed"] += 1
                    elif event.get("monsterType") == "RIFTHERALD":
                        aggregated_stats["blue" if event["killerTeamId"] == 100 else "red"]["herald_kills"] += 1
                    elif event.get("monsterType") == "DRAGON":
                        aggregated_stats["blue" if event["killerTeamId"] == 100 else "red"]["dragon_kills"] += 1
                    elif event.get("monsterType") == "BARON_NASHOR":
                        aggregated_stats["blue" if event["killerTeamId"] == 100 else "red"]["baron_kills"] += 1
                elif event.get("type") == "BUILDING_KILL":
                    if event.get("buildingType") == "TOWER_BUILDING":
                        victim_id = event.get("teamId")
                        if victim_id:
                            if victim_id == 200:
                                aggregated_stats["blue"]["tower_kills"] += 1
                            else:
                                aggregated_stats["red"]["tower_kills"] += 1
                    elif event.get("buildingType") == "INHIBITOR_BUILDING":
                        victim_id = event.get("teamId")
                        if victim_id:
                            if victim_id == 200:
                                aggregated_stats["blue"]["inhibitor_kills"] += 1
                            else:
                                aggregated_stats["red"]["inhibitor_kills"] += 1
                elif event.get("type") == "WARD_PLACED":
                    creator_id = event.get("creatorId")
                    if creator_id:
                        if creator_id <= 5:
                            aggregated_stats["blue"]["wards_placed"] += 1
                        else:
                            aggregated_stats["red"]["wards_placed"] += 1
                elif event.get("type") == "WARD_KILL":
                    killer_id = event.get("killerId")
                    if killer_id:
                        if killer_id <= 5:
                            aggregated_stats["blue"]["wards_killed"] += 1
                        else:
                            aggregated_stats["red"]["wards_killed"] += 1

        # append aggregated stats for the current minute
        stats_per_minute.append(aggregated_stats)

    # prepare the result dictionary
    result = {
        "winning_team": winning_team,
        "first_blood": first_blood,
        "stats_per_minute": stats_per_minute
    }

    return result


def get_match_champions(match_id, riot_api: RiotAPI):
    # retrieve match details and print the champion names of each participant in order
    base_url = f"https://americas.api.riotgames.com/lol/match/v5/matches/{match_id}"
    match_data = riot_api.make_request(base_url)

    # extract participant information from the 'info' field
    participants = match_data["info"]["participants"]

    # split into blue and red teams based on order
    blue_team_champs = [participant["championId"] for participant in participants[:5]]
    red_team_champs = [participant["championId"] for participant in participants[5:]]

    return blue_team_champs, red_team_champs

def process_match_ids(batch_size=50, filename="match_ids.json", output="match_info.json", riot_api: RiotAPI = None):
    # process match ids in batches, retrieve match information, and save the results
    match_ids_filepath = os.path.join("data", filename)
    match_info_filepath = os.path.join("data", output)

    # ensure data directory exists
    os.makedirs("data", exist_ok=True)

    # load match ids from file
    if not os.path.exists(match_ids_filepath):
        print(f"file {match_ids_filepath} does not exist. exiting.")
        return

    with open(match_ids_filepath, "r") as file:
        match_ids = json.load(file)

    # load existing match information if the file exists
    match_info = []
    if os.path.exists(match_info_filepath):
        with open(match_info_filepath, "r") as file:
            match_info = json.load(file)

    # process match ids in batches
    while match_ids:
        batch = match_ids[:batch_size]  # select the first batch
        match_ids = match_ids[batch_size:]  # update the remaining list

        batch_match_info = []

        for match_id in batch:
            # call process_match_timeline to get match stats
            match_data = process_match_timeline(match_id, riot_api)
            if not match_data:
                continue  # skip if timeline processing fails

            # get champion ids for blue and red teams
            blue_champs, red_champs = get_match_champions(match_id, riot_api)

            # add champion ids to match data
            match_data["blue_team_champions"] = blue_champs
            match_data["red_team_champions"] = red_champs
            match_data["match_id"] = match_id

            # append match data to the batch list
            batch_match_info.append(match_data)
            print(".", end="")
        print()

        # save the batch results to the match info file
        match_info.extend(batch_match_info)
        with open(match_info_filepath, "w") as file:
            json.dump(match_info, file, indent=4)

        # update the match ids file to remove processed ids
        with open(match_ids_filepath, "w") as file:
            json.dump(match_ids, file, indent=4)

        print(f"processed and saved batch of {len(batch)} matches. remaining: {len(match_ids)}. current in data: {len(match_info)}")

        # rotate api key
        riot_api.rotate_api_key()

    print("all match ids processed.")

def load_ids_to_db(json_path="data/match_ids.json", db_path="data/match_ids.db"):
    # read json file and load ids into a set
    with open(json_path, "r") as file:
        ids = set(json.load(file))

    # connect to sqlite database (or create it if it doesn't exist)
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # create a table to store unique ids if it doesn't already exist
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS match_ids (
            id TEXT PRIMARY KEY
        )
    ''')

    # insert each id into the database
    cursor.executemany("INSERT OR IGNORE INTO match_ids (id) VALUES (?)", [(id,) for id in ids])

    # commit changes and close connection
    conn.commit()
    conn.close()

def split_json_list(file_path, n):
    # load the json file
    with open(file_path, "r") as file:
        data = json.load(file)

    # check if the input is a list
    if not isinstance(data, list):
        raise ValueError("the json file must contain a list.")

    # calculate split sizes
    total_items = len(data)
    split_size = math.ceil(total_items / n)

    # split the list
    splits = [data[i:i + split_size] for i in range(0, total_items, split_size)]

    # write the splits into separate files
    output_dir = os.path.join(os.path.dirname(file_path), "splits")
    os.makedirs(output_dir, exist_ok=True)

    for i, split in enumerate(splits, start=1):
        output_file = os.path.join(output_dir, f"match_ids{i}.json")
        with open(output_file, "w") as out_file:
            json.dump(split, out_file, indent=4)
        print(f"created: {output_file}")

def thread_worker(thread_id, api_keys):
    riot_api = RiotAPI(api_keys)
    process_match_ids(filename=f"splits/match_ids{thread_id}.json", output=f"results/match_info{thread_id}.json", riot_api=riot_api)

def run_threads(num_threads=2):
    threads = []

    chunk_size = math.ceil(len(API_KEYS) / num_threads)
    thread_apis = [API_KEYS[i * chunk_size:(i + 1) * chunk_size] for i in range(num_threads)]

    # create threads
    for i in range(num_threads):
        thread = threading.Thread(target=thread_worker, args=(i + 1, thread_apis[i]))
        threads.append(thread)

    # start threads
    for thread in threads:
        thread.start()

    # wait for all threads to finish
    for thread in threads:
        thread.join()

    print("all threads have completed their tasks.")

# match_id = "NA1_5150259672"
# https://www.leagueofgraphs.com/match/na/5150259672
#


# load_ids_to_db()
# process_divisions()
riot_api = RiotAPI(API_KEYS)

# print_event_types("NA1_5150259672")
# split_json_list("data/match_ids.json", 3)
run_threads(3)
# process_match_timeline("NA1_5150259672")