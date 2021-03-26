from re import DEBUG
from numpy.lib.function_base import extract
import pretty_midi as pm
import matplotlib.pyplot as plt
import glob
import numpy as np 



class Visualizer:
    def __init__(self, midi_data,name = "midi file"):
        self.midi_data = midi_data
        self.name = name

    def show_midi_notes(self, DEBUG = False):
        n = len(self.midi_data.instruments)
        if DEBUG:
            print("Number of instruments : ", n)
        fig, axs = plt.subplots(n,1, figsize=(15, 10), facecolor='w', edgecolor='k')
        fig.subplots_adjust(hspace = .5, wspace=.1)
        if n>1:
            axs = axs.ravel()
            for j in range(n):
                notes = self.midi_data.instruments[j].get_piano_roll()
                axs[j].imshow(notes,cmap="cividis",aspect='auto')
                axs[j].set_title("Track : {} | Instrument : {}".format(self.name,self.midi_data.instruments[j].name))
                axs[j].set_ylim((0,128))
                axs[j].set_xlabel("Time")
                axs[j].set_ylabel("Pitch")
            plt.show()
        else:
            notes = self.midi_data.instruments[0].get_piano_roll()
            plt.imshow(notes,cmap="cividis",aspect='auto')
            plt.title("Track : {} | Instrument : {}".format(self.name,self.midi_data.instruments[0].name))
            plt.ylim((0,128))
            plt.xlabel("Time")
            plt.ylabel("Pitch")
            plt.show()

    def show_f0_velocity(self,frame_rate = 2500, DEBUG = False):
        n = len(self.midi_data.instruments)
        if DEBUG:
            print("Number of instruments : ", n)
        for instrument_data in self.midi_data.instruments:
            notes = instrument_data.get_piano_roll(frame_rate)
            pitches, loudness = self.extract_f0_loudness(notes)
            x = np.array([i/frame_rate for i in range(notes.shape[1])])


            fig, axs = plt.subplots(2,1, figsize=(15, 10), facecolor='w', edgecolor='k')
            fig.subplots_adjust(hspace = .5, wspace=.1)

            axs[0].plot(x,pitches)
            axs[0].set_title("Track : {} | Instrument : {}".format(self.name,instrument_data.name))
            axs[0].set_ylim((0,128))
            axs[0].set_xlabel("Time (s)")
            axs[0].set_ylabel("Pitch")

            axs[1].plot(x,loudness)
            axs[1].set_title("Track : {} | Instrument : {}".format(self.name,instrument_data.name))
            axs[1].set_ylim((0,128))
            axs[1].set_xlabel("Time (s)")
            axs[1].set_ylabel("Loudness")
        plt.show()


    def extract_f0_loudness(self, notes):
        pitches = np.argmax(notes, axis = 0)
        loudness = np.max(notes, axis = 0)
        return pitches, loudness







