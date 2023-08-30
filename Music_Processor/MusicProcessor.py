from pathlib import Path
import os
import numpy as np
import collections
import pretty_midi
import pandas as pd
from matplotlib import pyplot as plt
from typing import Dict, List, Optional, Sequence, Tuple
from magenta.models.performance_rnn import performance_sequence_generator
from magenta.models.shared import sequence_generator_bundle
from magenta.music import constants
from magenta.music.protobuf import generator_pb2
from magenta.music.protobuf import music_pb2
import mido
from magenta.music import midi_io
from magenta.music import sequences_lib
from note_seq import midi_io as Nmid

class MusicProcessor:
    def __init__(self, key_order=['pitch', 'step', 'duration']):
        self.key_order = key_order

    def get_name(self,midi_file):
        pm = pretty_midi.PrettyMIDI(midi_file)
        instrument = pm.instruments[0]
        instrument_name = pretty_midi.program_to_instrument_name(instrument.program)
        return instrument_name

    def preprocess(self, midi_file):
        raw_notes = self.midi_to_notes(midi_file)
        sample_notes = self.convert_to_sample_notes(raw_notes)
        return sample_notes

    def midi_to_notes(self,midi_file_path: str) -> pd.DataFrame:
        pm = pretty_midi.PrettyMIDI(midi_file_path)
        instrument = pm.instruments[0]
        notes = collections.defaultdict(list)

        # Sort the notes by start time
        sorted_notes = sorted(instrument.notes, key=lambda note: note.start)
        prev_start = sorted_notes[0].start

        for note in sorted_notes:
            start = note.start
            end = note.end
            notes['pitch'].append(note.pitch)
            notes['start'].append(start)
            notes['end'].append(end)
            notes['step'].append(start - prev_start)
            notes['duration'].append(end - start)
            prev_start = start

        return pd.DataFrame({name: np.array(value) for name, value in notes.items()})

    def plot_piano_roll(self,notes: pd.DataFrame, count: Optional[int] = None):
        if count:
            title = f'First {count} notes'
        else:
            title = f'Whole track'
            count = len(notes['pitch'])
            
            plt.figure(figsize=(20, 4))
            plot_pitch = np.stack([notes['pitch'], notes['pitch']], axis=0)
            plot_start_stop = np.stack([notes['start'], notes['end']], axis=0)
            
            plt.plot(plot_start_stop[:, :count], plot_pitch[:, :count], color="b", marker=".")
            plt.xlabel('Time [s]')
            plt.ylabel('Pitch')
            _ = plt.title(title)

    def convert_to_sample_notes(self, raw_notes):
        sample_notes = np.stack([raw_notes[key] for key in self.key_order], axis=1)
        
        return sample_notes

    def notes_to_midi(self, notes: pd.DataFrame, out_file: str, instrument_name: str,
                  velocity: int = 100) -> pretty_midi.PrettyMIDI:

        print(out_file)
        pm = pretty_midi.PrettyMIDI()
        instrument = pretty_midi.Instrument(
        program=pretty_midi.instrument_name_to_program(
            instrument_name))
        
        for i, note in notes.iterrows():
            
            note = pretty_midi.Note(velocity=velocity, pitch=int(note['pitch']),
                                    start=float(note['start']), end=float(note['end']))
            instrument.notes.append(note)

        pm.instruments.append(instrument)
        pm.write(out_file)
        return pm
    
    def preprocess_magenta(self, midi_file):
        """
        Convert the MIDI file to a performance event sequence suitable for Performance RNN input.
        """

        midi_data = mido.MidiFile(midi_file)


        sequence = midi_io.midi_to_note_sequence(midi_data)

        # Convert NoteSequence to performance events
        performance_events = performance_sequence_generator.get_performance(
            sequence, start_step=0, num_steps=100, num_velocity_bins=0
        ).quantize_note_times(1).events

        return performance_events 
    
    def notes_to_midi_magenta(self, events, out_file):
        """
        Convert the performance events back to a MIDI file.
        """
        # Convert events to a performance NoteSequence
        performance_sequence = performance_sequence_generator.get_performance_from_events(
            events
        ).to_sequence()

        # Convert the performance NoteSequence back to MIDI
        midi_data = midi_io.note_sequence_to_midi(performance_sequence)

        # Save the MIDI data to a file
        with open(out_file, 'wb') as f:
            f.write(midi_data)

    def trim_to_last_n_seconds(sequence, n):
        """Trim a NoteSequence to only include the last n seconds."""
        end_time = sequence.total_time
        start_time = max(0, end_time - n)
        
        # Create a new sequence for the trimmed part
        trimmed_sequence = music_pb2.NoteSequence()
        
        # Copy the notes that fall into the last n seconds
        for note in sequence.notes:
            if note.start_time >= start_time and note.end_time <= end_time:
                new_note = trimmed_sequence.notes.add()
                new_note.MergeFrom(note)  # copy note to new sequence
                
        # Copy other fields (time signatures, tempos, etc.)
        for field in ['tempos', 'time_signatures', 'key_signatures']:
            for item in getattr(sequence, field):
                if item.time >= start_time:
                    getattr(trimmed_sequence, field).add().MergeFrom(item)
                    
        # Set total time
        trimmed_sequence.total_time = n

        return trimmed_sequence
    
    def generation_offset(sequence,offset):
        for note in sequence.notes:
            note.start_time -= offset
            note.end_time -= offset
        return sequence