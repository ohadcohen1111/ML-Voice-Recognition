import pandas as pd
import re
import scipy.stats as stats
from scipy.io import wavfile
import numpy as np
import os

columns=['nobs', 'mean', 'skew', 'kurtosis', 'median', 'mode', 'std', 'low', 'peak', 'q25', 'q75', 'iqr',
 'user_name', 'sample_date', 'age_range', 'pronunciation', 'gender']

myData = pd.DataFrame(columns=columns)



# def get_metadata(readme_file):
#
#     #define variables in case startswith does not work:
#     gender, age_range, pronunciation = 'not specified', 'not specified', 'not specified'
#     for line in open(readme_file):
#         if line.startswith("Gender:"):
#             gender = line.split(":")[1].strip()
#         elif line.startswith("Age Range:"):
#             age_range = line.split(":")[1].strip()
#         elif line.startswith("Pronunciation dialect:"):
#             pronunciation = line.split(":")[1].strip()
#     return gender, age_range, pronunciation


def get_features(frequencies):

    print("\nExtracting features ")
    nobs, minmax, mean, variance, skew, kurtosis = stats.describe(frequencies)
    median = np.median(frequencies)
    mode = stats.mode(frequencies).mode[0]
    std = np.std(frequencies)
    low,peak = minmax
    q75,q25 = np.percentile(frequencies, [75 ,25])
    iqr = q75 - q25
    return nobs, mean, skew, kurtosis, median, mode, std, low, peak, q25, q75, iqr


# def get_date(sample_name):
#
#     try:
#         date = pattern_date.search(sample_name).group()
#     except AttributeError:
#         date = '20000000'
#     return date


def get_frequencies():
    '''
    extract list of dominant frequencies in sliding windows of duration defined by 'step' for each of the 10 wav files and return an array
    frequencies_lol: list of lists
    every item in this list will contain 10 lists corresponding to each of the 10 wav files in every sample
    and the lists within the list will contain a range of filtered frequencies corresponding to sliding windows within each wav file
    '''
    frequencies_lol = []
    # for wav_file in os.listdir(sample_wav_folder):
    #     rate, data = wavfile.read(os.path.join(sample_wav_folder, wav_file))
    wav_file = 'b0019.wav'  # Noise at 50Hz #check plot_frequency
    # wav_file = '/home/vitalv/voice-gender-classifier/raw/anonymous-20100621-cdr/wav/a0166.wav'
    rate, data = wavfile.read(wav_file)
        # get dominating frequencies in sliding windows of 200ms
    step = rate / 5  # 3200 sampling points every 1/5 sec
    window_frequencies = []
    step = int(step)
    for i in range(0, len(data), step):
    # i = 0
    # step = int(step)
    # while i < len(data):
        ft = np.fft.fft(data[i:i + step])
        freqs = np.fft.fftfreq(len(ft))  # fftq tells you the frequencies associated with the coefficients
        imax = np.argmax(np.abs(ft))
        freq = freqs[imax]
        freq_in_hz = abs(freq * rate)
        window_frequencies.append(freq_in_hz)
        filtered_frequencies = [f for f in window_frequencies if 20 < f < 300 and not 46 < f < 66]
        # I see noise at 50Hz and 60Hz. See plots below
    frequencies_lol.append(filtered_frequencies)
    frequencies = [item for sublist in frequencies_lol for item in sublist]
    return frequencies


# for i in range(n_samples):
if __name__ == '__main__':
    # get the path to the wav files (.raw/wav) and to the README file (.raw/etc/README)
    # sample = sorted(samples)[i]
    # sample_folder = os.path.join(raw_folder, sample)
    # sample_wav_folder = os.path.join(sample_folder, 'wav')
    # readme_file = os.path.join(sample_folder, 'etc', 'README')

    # get the information from the readme file: gender, age_range, pronunciation
    # date = get_date(sample)
    # user_name = get_user_name(sample)
#     if os.path.isfile(readme_file):
#         gender, age_range, pronunciation = get_metadata(readme_file)
# gender, age_range, pronunciation = homogenize_format(gender, age_range, pronunciation)

# Read and extract the information from the wav files:
# if os.path.isdir(sample_wav_folder):  # some of the samples don't contain a wav folder (Ex: 'LunaTick-20080329-vf1')
    frequencies = get_frequencies()
    if len(frequencies) > 8:
        # for some of the files (ex: Aaron-20130527-giy)
        # I only recover frequencies of 0.0 (even if I don't split in chunks) which is not integrated into my lol and frequencies is empty
        nobs, mean, skew, kurtosis, median, mode, std, low, peak, q25, q75, iqr = get_features(frequencies)
        sample_dict = {'nobs': nobs/1000, 'mean': mean/1000, 'skew': skew/1000, 'kurtosis': kurtosis/1000,
                       'median': median/1000, 'mode': mode/1000, 'std': std/1000, 'low': low/1000,
                       'peak': peak/1000, 'q25': q25/1000, 'q75': q75/1000, 'iqr': iqr/1000,
                       }
                       # 'user_name': user_name, 'sample_date': date,
                       # 'age_range': age_range, 'pronunciation': pronunciation,
                       # 'gender': gender}
        # print
        # "\nappending %s sample %s : %s" % (gender, sample, sample_dict)

        # Save to my pandas dataframe
    myData = pd.Series(sample_dict)

# and store it to a file
    myData.to_csv('myData_filtered.csv')
