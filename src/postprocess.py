"""
This module converts a numpy array (one bar) back into a MIDI file.
It loads a numpy array that represents note events in a 10 (families) x 48 (grid locations) array.
For each non-empty cell, a note_on event is created using its family (mapped to a specific MIDI note)
and its onset time computed from a custom division grid.
The resulting MIDI messages are printed to the console before saving the MIDI file.
"""

import os
import numpy as np
from mido import Message, MidiFile, MidiTrack

TOTAL_NUM_FAMILIES = 10
TOTAL_TIME_LOCATIONS = 192

def get_division_grid() -> list:
    """
    Recreates the custom division grid used in the preprocessing.
    It combines sixteenth note subdivisions and eighth-note triplet subdivisions,
    and then appends 4.0 to account for the bar’s end.

    Returns:
         list of float: Sorted grid values.
    """
    # sixteenth notes: 32 subdivisions in 4 beats (endpoint not included)
    sixteenth_divs = np.linspace(0, 4, 128, endpoint=False)
    # eighth note triplets: 24 subdivisions in 4 beats (endpoint not included)
    eighth_triplet_divs = np.linspace(0, 4, 96, endpoint=False)
    grid = sorted(set(sixteenth_divs).union(set(eighth_triplet_divs)))
    grid.append(4.0)
    return grid

# We use one representative MIDI note for each family.
# These values are selected from the original mapping in gdrm/preprocess.py.
# For example, family 1 always uses note 36; family 2 uses note 38; etc.
family_to_midi = {
    1: 36,
    2: 38,
    3: 48,
    4: 45,
    5: 43,
    6: 46,
    7: 42,
    8: 49,
    9: 57,
    10: 51
}

def create_midi_from_array(npy_file: str, output_midi: str, ticks_per_beat: int = 480) -> None:
    """
    Reads a numpy array from a file, then converts the note event data back into a MIDI file.
    Each nonempty cell (value > -1) in the array is assumed to represent a note event.

    The onset time is computed by taking the column index's position on a custom grid
    and multiplying by ticks_per_beat. The velocity is recovered by adding 1 (since the array
    was initialized to -1 so that a note event with velocity v appears as v-1 in the array).

    Parameters:
         npy_file (str): Path to the numpy file (one bar array) to read.
         output_midi (str): Path to the output MIDI file.
         ticks_per_beat (int, optional): Ticks per beat value; defaults to 480.
    """
    # Load the numpy array (should be shape: 10 x 48)
    try:
        bar_array = np.load(npy_file)
    except Exception as e:
        print(f"Error loading numpy file: {e}")
        return

    # Retrieve the custom rhythmic grid.
    grid = get_division_grid()
    if len(grid) < TOTAL_TIME_LOCATIONS:
        print("Warning: grid length is less than total time locations. Check grid generation!")

    events = []
    # Loop over each column (time grid) in order.
    # Each column represents a grid subdivision; we compute the absolute tick
    # by multiplying the grid value (in beats) with ticks_per_beat.
    for col in range(TOTAL_TIME_LOCATIONS):
        # Compute the onset time in ticks for this grid index.
        # (The grid was used to quantize onset times into beats.)
        onset_tick = int(round(grid[col] * ticks_per_beat))
        for row in range(TOTAL_NUM_FAMILIES):
            cell_value = bar_array[row, col]
            if cell_value > -1:  # cell contains a note event
                # Recover velocity (the original note velocity was added to -1)
                velocity = int(cell_value + 1)
                midi_note = family_to_midi.get(row+1)  # convert 0-indexed row to family number
                if midi_note is None:
                    continue
                # Append the event as a tuple: (absolute_tick, midi_note, velocity)
                events.append((onset_tick, midi_note, velocity))

    # Sort events by onset time so we can convert them to delta times.
    events.sort(key=lambda x: x[0])

    # Convert the sorted list of events (with absolute ticks) into MIDI messages.
    midi_messages = []
    prev_tick = 0
    for abs_tick, midi_note, velocity in events:
        delta = abs_tick - prev_tick
        msg = Message("note_on", note=midi_note, velocity=velocity, time=delta)
        midi_messages.append(msg)
        prev_tick = abs_tick

    # Print MIDI messages to the console for debugging.
    print("Generated MIDI messages:")
    for msg in midi_messages:
        print(msg)

    # Create MIDI file and track; add the note messages.
    mid = MidiFile(ticks_per_beat=ticks_per_beat)
    track = MidiTrack()
    mid.tracks.append(track)

    for msg in midi_messages:
        track.append(msg)

    # Save the MIDI file.
    try:
        mid.save(output_midi)
        print(f"MIDI file successfully saved as '{output_midi}'.")
    except Exception as e:
        print(f"Error saving MIDI file: {e}")

def main():
    # Example: use the numpy file for bar 1 from the "bar_arrays" folder.
    npy_file = os.path.join("bar_arrays/drummer1/session1", "1_funk_80_beat_4-4_bar_4.npy")
    npy_file = "generated_output.npy"
    output_midi = "generated_output.mid"
    create_midi_from_array(npy_file, output_midi)

if __name__ == "__main__":
    main()
