from audio2midi import Audio2MidiConverter
from extract_f0_confidence_loudness import Extractor
from txt2contours import Txt2Contours



import sys
sys.path.insert(0,'..')
from midiConverter import Converter

from tqdm import tqdm
import os
import numpy as np
import sounddevice as sd
import soundfile as sf
import matplotlib.pyplot as plt
import librosa as li
import seaborn as sns

import pandas as pd
import pretty_midi as pm
import note_seq
from note_seq.protobuf import music_pb2
import matplotlib.pyplot as plt






class Eval:
    def __init__(self):
        pass

    def evaluate(self, midi_file, text_file, sampling_rate=48000, block_size=480, verbose=False):

        # From text file
        t2c = Txt2Contours()
        time_text, frequency_text, loudness_text = t2c.process(text_file, sampling_rate/block_size)


        # From midi file : 

        c = Converter()
        midi_data = pm.PrettyMIDI(midi_file)
        time_gen, frequency_gen, loudness_gen = c.midi2time_f0_loudness(midi_data, sampling_rate/block_size)
        
        frequency_gen = li.core.midi_to_hz(frequency_gen)
        loudness_gen = loudness_gen / np.max(loudness_gen)
        # want to erase really quiet notes
        loudness_threshold = 0.20
        frequency_gen = frequency_gen * (loudness_gen>loudness_threshold)


        # 0 padding (text file are shorter since they do not consider last silence)
        frequency_text = np.concatenate((frequency_text, np.zeros(time_gen.shape[0]-time_text.shape[0])))
        loudness_text = np.concatenate((loudness_text, np.zeros(time_gen.shape[0]-time_text.shape[0])))


        diff_f0 = np.abs(frequency_gen - frequency_text)
        diff_loudness = np.abs(loudness_text - loudness_gen)

        # np.concatenate((frequencyHertz, np.zeros(np.abs(time_text.shape[0]-time_gen.shape[0]))))

        # compute difference and score:

        #diff = np.abs(frequency_text-new_freq_gen)
        
        score = np.mean(diff_f0) + np.mean(diff_loudness)


        if verbose:
            ax1 = plt.subplot(221)        
            ax1.plot(time_gen, frequency_text, label = "text")
            ax1.plot(time_gen, frequency_gen, label = "midi" )
            ax1.set_title("f0 comparison")

            ax2 = plt.subplot(222)
            ax2.plot(time_gen, loudness_text, label = "text")
            ax2.plot(time_gen, loudness_gen/np.max(loudness_gen), label = "midi" )
            ax2.set_title('Loudness comparison')

            ax3 = plt.subplot(223)
            ax3.plot(time_gen, diff_f0, label = "text")
            ax3.set_title('f0 differences')

            ax4 = plt.subplot(224)
            ax4.plot(time_gen, diff_loudness, label = "text")
            ax4.set_title('Loudness differences')



            plt.legend()
            plt.show()

        return score



if __name__ == '__main__':

    save_path = "midi-generated-files/"
    midi_file = save_path + "violin(from-audio)-thC0.4-thM0.01.mid"
    txt_file = "violin.txt"
    e = Eval()
    score = e.evaluate(midi_file, txt_file, verbose=True)
    print("Total score ", score)
