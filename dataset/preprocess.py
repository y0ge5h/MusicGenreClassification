import os
import librosa
import math
import json

DATA_SET_PATH = '/Users/Yogesh/Downloads/genres/'
JSON_PATH = 'dataset/data.json'
SAMPLE_RATE = 22050
DURATION = 30  # secs
SAMPLES_PER_TRACK = SAMPLE_RATE * DURATION  # 661500


def save_mfcc(dataset_path, json_path, n_mfcc=13, hop_length=512, n_fft=2048, n_segments=5):
    # bulid dictionary to load data
    data = {
        'mapping': [],
        'label': [],
        'mfcc': []
    }

    num_samples_per_segments = int(SAMPLES_PER_TRACK/n_segments) 
    expected_no_mfcc_vectors_per_segment = math.ceil(num_samples_per_segments / hop_length)
    # loop through all genres
    for i, (dirpath, dirnames, filenames) in enumerate(os.walk(dataset_path)):

        if dirpath is not dataset_path:
            # save the symantic label > genres/blues > ['genres','blues'] > idx = -1 , value = 'blues'
            label = dirpath.split('/')[-1]
            data['mapping'].append(label)
            print(f"\n processing {label}")
            # process files for genre
            for f in filenames:
                filepath = os.path.join(dirpath, f)
                signal, sr = librosa.load(filepath, sr=SAMPLE_RATE)
                print(signal.shape)
                # process segments
                for s in range(n_segments):
                    start_sample = num_samples_per_segments * s
                    finish_sample = start_sample + num_samples_per_segments

                    # extract the mfccs
                    mfcc = librosa.feature.mfcc(signal[start_sample:finish_sample], sr=sr,
                                                n_mfcc=13, hop_length=512, n_fft=2048)
                    mfcc = mfcc.T
                    # store mfcc for segment if it has expected length
                    if len(mfcc) == expected_no_mfcc_vectors_per_segment:
                        data['mfcc'].append(mfcc.tolist())
                        data['label'].append(i-1)
                        print(f"{f},{s+1}")

    with open(json_path, 'w') as jf:
        json.dump(data, jf, indent=4)
        print(f"file {json_path} created successfully...")


if __name__ == '__main__':
    save_mfcc(DATA_SET_PATH, JSON_PATH, n_segments=10)