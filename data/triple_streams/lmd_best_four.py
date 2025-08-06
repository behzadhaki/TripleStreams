import math
from multiprocessing import Pool
import os
import pickle, bz2
from tqdm import tqdm
import sys



def load_dataset(file_path):
    with bz2.BZ2File(file_path, 'rb') as f:
        dataset = pickle.load(f)
    return dataset

# Sorts the list of hvo sequences and finds the one with the lowest amount of hits, the highest amount of hits, 1/4 the max of hits, and 3/4 max of hits
# Input: dictionary from LMD track
# Output: new dictionary of same structure, but with split_n_bar_phrases updated with the best four hvo sequences
def filter_best_four(track_dict):
    new_dict = {}
    sorting_list = []
    splits = track_dict['split_n_bar_phrases']
    phrases = []
    
    for i, hvo in enumerate(splits):
        if sum(sum(hvo.hits)) > 0:
            sorting_list.append([i, sum(sum(hvo.hits))])
    
    sorted_hvo = sorted(sorting_list, key=lambda x: x[1])
    
    if len(sorted_hvo) > 8:
        n = len(sorted_hvo)
        #indices = [int(math.floor(n * i / 8)) for i in range(8)]
        indices = [int(math.floor(n / 2 + (n / 2) * i / 8)) for i in range(8)]
        indices = [min(idx, n - 1) for idx in indices]

        for idx in indices:
            phrases.append(splits[sorted_hvo[idx][0]])

        new_dict['split_n_bar_phrases'] = phrases
        new_dict['filenames'] = track_dict['filenames']
        new_dict['hvo_sequences'] = track_dict['hvo_sequences']
        new_dict['compiled_single_hvos'] = track_dict['compiled_single_hvos']
    else:
        print(f"Not enough hvo sequences to filter. Found: {len(sorted_hvo)}")
    
    return new_dict


# Goes through each track in the dictionary and runs the filter_best_four function on each
# Input: Dictionary (first level) with track name values
# output: New Dictionary with new dictionaries for each track with updated best hvo values
def get_hvo_sequence_from_track_dictionary(dict):
    new_dict = {}
    best_four_dict = {}
    for track in dict['AllAvailableStreams']:
        print(track)
        track_dict = dict['AllAvailableStreams'][track]
        best_four_dict = filter_best_four(track_dict)
        if 'split_n_bar_phrases' in best_four_dict:
            new_dict[track] = best_four_dict
    return new_dict
    

def run_all(pkl_path):
 

    print("Processing:", pkl_path)
    try:
        dataset = load_dataset(pkl_path)
        filtered_dict = get_hvo_sequence_from_track_dictionary(dataset)

        base_name = os.path.basename(pkl_path).replace('.pkl.bz2', '')
        output_name = f"{base_name}_filtered.pkl.bz2"
        output_path = os.path.join('data/triple_streams/split_2bars/lmd_top_four', output_name)

        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        # Save the filtered dictionary
        with bz2.BZ2File(output_path, 'wb') as f:
            pickle.dump({'AllAvailableStreams': filtered_dict}, f)

        print(f"Saved filtered file: {output_path}")
        return True

    except Exception as e:
        print(f"Error processing {pkl_path}: {e}")
        return False


def main(dataset_dir):
    pkl_files = [
        os.path.join(dataset_dir, f)
        for f in os.listdir(dataset_dir)
        if f.endswith('.pkl.bz2')
    ]

    with Pool(processes=5) as pool:
        list(tqdm(pool.imap(run_all, pkl_files), total=len(pkl_files)))

if __name__ == '__main__':
    if len(sys.argv) > 2:
        print("Please only pass the data directory")
        sys.exit(1)
    main(sys.argv[1])