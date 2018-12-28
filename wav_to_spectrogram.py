## Convert audio WAV files into Mel-Spectrograms
## Tool: https://github.com/timsainb/python_spectrograms_and_inversion/blob/master/Python-Spectrograms-MFCC-and-Inversion.ipynb

from scipy.io import wavfile
import mel_lib
import numpy as np

def extract_mel_spec(waves, samplingFreq=44100):
    
    fft_size = 2048
    step_size = 1024
    n_mel_freq_components = 128
    shorten_factor = 1 # how much should we compress the x-axis (time)
    start_freq = 0 # Hz # What frequency to start sampling our melS from 
    end_freq = samplingFreq / 2 # Hz # What frequency to stop sampling our melS from 
    spec_thresh = 4
    
    wav_spectrogram = mel_lib.pretty_spectrogram(waves.astype('float64'), fft_size = fft_size, 
                                   step_size = step_size, log = True, thresh = spec_thresh)
    # Generate the mel filters
    mel_filter, mel_inversion_filter = mel_lib.create_mel_filter(fft_size = fft_size,
                                                            n_freq_components = n_mel_freq_components,
                                                            start_freq = start_freq,
                                                            end_freq = end_freq, samplerate=samplingFreq)

    mel_spec = mel_lib.make_mel(wav_spectrogram, mel_filter, shorten_factor = shorten_factor)
    return mel_spec.astype('float32')

if __name__ == '__main__':
    
    samplingFreq, mySound = wavfile.read('PATH_TO_A_WAV_AUDIO_FILE')
    mel_spec = extract_mel_spec(mySound)
    np.save('PATH_TO_SAVE_SPECTROGRAM', mel_spec)