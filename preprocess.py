#!/usr/bin/env python3
"""
This module processes MIDI files for drummer groove data. Besides extracting tempo,
time signature, note events, and specific control change events, it now supports a pipeline mode.
It recursively walks a dataset, analyzes each MIDI filename, processes it, and saves
each bar as a numpy array to a directory called "bar_arrays" whose subdirectory structure mirrors the original dataset.

Filename pattern assumed:
    idx_genre_bpm_type_tsnumerator-tsdenominator.mid

Each numpy array is saved as:
    {original_filename}_bar_{barnum}.npy
"""

TOTAL_NUM_FAMILIES = 10
TOTAL_TIME_LOCATIONS = 192

import json  # no longer used but kept in case for debugging
import os
import numpy as np
from typing import Tuple, Any, Dict, List, Optional
from mido import MidiFile, Message, tempo2bpm

def get_division_grid() -> List[float]:
    """
    Constructs and returns the custom grid of rhythmic subdivisions.

    The grid is constructed using sixteenth note divisions and eighth-note triplet divisions,
    then merging these values and appending 4.0 to account for the bar-end.

    Returns:
        List[float]: Sorted list of grid subdivision values.
    """
    sixteenth_divs = np.linspace(0, 4, 128, endpoint=False)
    eighth_triplet_divs = np.linspace(0, 4, 96, endpoint=False)
    grid = sorted(set(sixteenth_divs).union(set(eighth_triplet_divs)))
    grid.append(4.0)
    return grid

def closest_value(target: float, values: Optional[List[float]] = None) -> Tuple[float, int]:
    """
    Returns the value in 'values' that is closest to the 'target', and its index.

    If no list is provided, it uses the custom division grid from get_division_grid().

    Parameters:
        target (float): The value to compare against.
        values (List[float], optional): Candidate values; defaults to the custom division grid.

    Returns:
        Tuple[float, int]: The closest value and its index.
    """
    if values is None:
        values = get_division_grid()
    best_index = None
    best_value = None
    best_diff = float('inf')

    for idx, val in enumerate(values):
        diff = abs(val - target)
        if diff < best_diff:
            best_diff = diff
            best_value = val
            best_index = idx
    return best_value, best_index

def get_track_object(midi_file: str) -> Tuple[MidiFile, List[Message]]:
    """
    Loads a MIDI file and returns the MidiFile object along with its first track.

    Parameters:
        midi_file (str): Path to the MIDI file.

    Returns:
        Tuple[MidiFile, List[Message]]:
            - MidiFile: Object representation of the MIDI file.
            - List[Message]: The first track from the MIDI file.
    """
    midi = MidiFile(midi_file)
    if not midi.tracks:
        raise ValueError("The MIDI file does not contain any tracks.")
    return midi, midi.tracks[0]

def get_track_info(messages: List[Message]) -> Tuple[int, float, int, int, int, int]:
    """
    Extracts tempo and time signature information from the MIDI messages.

    Parameters:
        messages (List[Message]): List of MIDI messages from which metadata is extracted.

    Returns:
        Tuple containing:
            - tempo (int): Microseconds per beat (default: 50000).
            - bpm (float): Beats per minute (default: 120).
            - numerator (int): Time signature numerator (default: 4).
            - denominator (int): Time signature denominator (default: 4).
            - clocks_per_click (int): MIDI clocks per click (default: 24).
            - notated_32nd_notes_per_beat (int): Notated 32nd notes per beat (default: 8).
    """
    # Default meta event values
    tempo = 50000
    bpm = 120
    numerator = 4
    denominator = 4
    clocks_per_click = 24
    notated_32nd_notes_per_beat = 8

    for msg in messages:
        if msg.type == 'time_signature':
            numerator = msg.numerator
            denominator = msg.denominator
            clocks_per_click = msg.clocks_per_click
            notated_32nd_notes_per_beat = msg.notated_32nd_notes_per_beat
        elif msg.type == 'set_tempo':
            tempo = msg.tempo
            bpm = tempo2bpm(tempo)
    return tempo, bpm, numerator, denominator, clocks_per_click, notated_32nd_notes_per_beat

def process_track(track: List[Message], ticks_per_beat: int) -> Dict[str, Any]:
    """
    Processes MIDI track messages to extract note events, grouping them into bars,
    and computes normalized onset positions using a custom rhythmic grid.

    Parameters:
        track (List[Message]): List of MIDI messages.
        ticks_per_beat (int): Ticks per beat defined by the MIDI file.

    Returns:
        Dict[str, Any]: Dictionary mapping bar keys (e.g. 'bar_1') to their note events.
    """
    # Mapping of MIDI note numbers to families.
    note_family_mapping = {
        36: 1, 38: 2, 40: 2, 37: 2, 48: 3, 50: 3,
        45: 4, 47: 4, 43: 5, 58: 5, 46: 6, 26: 6,
        42: 7, 22: 7, 44: 7, 49: 8, 55: 8, 57: 9,
        52: 9, 51: 10, 59: 10, 53: 10
    }

    # Retrieve global tempo and time signature values using the first few messages.
    tempo, bpm, numerator, denominator, clocks_per_click, notated_32nd_notes_per_beat = get_track_info(track[:5])

    # Compute ticks per bar.
    ticks_per_bar = ticks_per_beat * (4 * numerator // denominator)

    # Retrieve the division grid once.
    grid = get_division_grid()

    # Dictionary to hold the organized bar-wise data.
    bars: Dict[str, Dict[str, List[Any]]] = {}
    abs_time = 0  # Running total of absolute ticks

    for msg in track:
        abs_time += msg.time  # msg.time is delta ticks

        # Process note_on events with nonzero velocity.
        if msg.type == 'note_on' and msg.velocity != 0:
            base_bar = abs_time // ticks_per_bar
            onset_in_bar = abs_time % ticks_per_bar
            onset_in_beats = onset_in_bar / ticks_per_beat

            # Compute normalized onset using the grid.
            normalized_onset, normalized_index = closest_value(onset_in_beats, values=grid)

            # If the normalized onset equals the bar-end, treat as beginning of next bar.
            if normalized_onset == 4.0:
                normalized_onset = 0.0
                onset_in_beats = 0.0
                # Re-calculate the normalized index for 0.0 using the grid.
                normalized_index = grid.index(0.0)
                base_bar += 1

            # Round values as needed.
            onset_in_beats = round(onset_in_beats, 2)
            normalized_onset = round(normalized_onset, 3)

            # Create bar key (1-indexed).
            bar_key = f"bar_{base_bar + 1}"
            family_index = note_family_mapping.get(msg.note, None)

            note_event = {
                "midi_note_number": msg.note,
                "velocity": msg.velocity,
                "onset_position_in_track": abs_time,
                "onset_position_in_bar": onset_in_bar,
                "onset_position_in_bar_by_beat": onset_in_beats,
                "normalized_onset_position_in_bar_by_beat": normalized_onset,
                "normalized_onset_index": normalized_index,
                "family_index": family_index
            }
            if bar_key not in bars:
                bars[bar_key] = {"notes": []}
            bars[bar_key]["notes"].append(note_event)

    return bars

def save_bar_arrays(bars: Dict[str, Any], output_dir: str, base_filename: str) -> None:
    """
    Converts each bar's note events into a numpy array and saves it as a .npy file.

    The array has dimensions:
        total_num_families x total_time_locations.

    Each note event is placed using its 0-indexed family index (row) and its normalized onset index (column).
    If multiple notes fall into the same cell, the velocity values are accumulated.

    The numpy array file is saved as:
         {base_filename}_bar_{barnum}.npy
    in the provided output_dir.

    Parameters:
        bars (Dict[str, Any]): Dictionary mapping bar keys to their note events.
        output_dir (str): Destination folder for bar arrays.
        base_filename (str): The base filename (without extension) of the original MIDI file.
    """
    total_num_families = TOTAL_NUM_FAMILIES
    total_time_locations = TOTAL_TIME_LOCATIONS

    # Ensure the output directory exists.
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    for bar_key, bar_data in bars.items():
        # Initialize an empty array for the bar.
        bar_array = np.full((total_num_families, total_time_locations), -1, dtype=np.int32)

        for note in bar_data["notes"]:
            family_index = note.get("family_index")
            normalized_index = note.get("normalized_onset_index")
            velocity = note.get("velocity")
            # Proceed only if family_index is defined.
            if family_index is not None:
                row = family_index - 1  # convert to 0-indexed
                col = normalized_index        # provided index from grid calculation
                # Accumulate velocity; start from -1, so add velocity and then adjust.
                bar_array[row, col] += velocity

        # Derive a numeric bar number from the bar key:
        # e.g., "bar_1" becomes "1"
        bar_num = bar_key.split("_")[-1]

        # Build output file name.
        out_filename = f"{base_filename}_bar_{bar_num}.npy"
        output_path = os.path.join(output_dir, out_filename)
        np.save(output_path, bar_array)

def process_midi_file(midi_file_path: str, dataset_root: str, output_root: str) -> None:
    """
    Processes a single MIDI file:
      - Loads and processes the track.
      - Parses and extracts the filename information.
      - Saves each bar’s note events as a numpy array into the output folder, under a directory structure
        mirroring the dataset.

    Parameters:
        midi_file_path (str): Full path to the MIDI file.
        dataset_root (str): Root folder of the dataset (used to create relative output paths).
        output_root (str): Root output folder for bar arrays ("bar_arrays").
    """

    # Parse the original filename and remove its extension.
    original_filename = os.path.splitext(os.path.basename(midi_file_path))[0]

    # (Optional) Parse the filename based on the assumed pattern:
    #   idx_genre_bpm_type_tsnumerator-tsdenominator.mid
    # In case you need to analyze these fields:
    try:
        parts = original_filename.split("_")
        idx = parts[0]
        genre = parts[1]
        bpm_field = parts[2]
        type_field = parts[3]
        time_sig = parts[4]  # expected to be like "4-4"
        # You can do further processing if needed.
    except Exception:
        # If parsing fails, just move on.
        pass

    try:
        midi_obj, track_obj = get_track_object(midi_file_path)
    except Exception as e:
        print(f"Error loading MIDI file {midi_file_path}: {e}")
        return

    bars = process_track(track_obj, midi_obj.ticks_per_beat)

    # Determine the relative folder of the midi file within dataset_root.
    rel_path = os.path.relpath(os.path.dirname(midi_file_path), dataset_root)
    # Build the output directory to mirror the dataset structure.
    output_dir = os.path.join(output_root, rel_path)
    save_bar_arrays(bars, output_dir, original_filename)

def process_dataset(dataset_root: str, output_root: str = "bar_arrays") -> None:
    """
    Walks through the dataset folder, processes each MIDI file found, and saves the corresponding bar numpy arrays.

    Parameters:
        dataset_root (str): The root directory of the MIDI dataset.
        output_root (str): The output directory where bar arrays will be saved.
    """
    for root, dirs, files in os.walk(dataset_root):
        for file in files:
            if file.lower().endswith(".mid"):
                midi_file_path = os.path.join(root, file)
                print(f"Processing {midi_file_path} ...")
                process_midi_file(midi_file_path, dataset_root, output_root)

def main() -> None:
    """
    Main entry point:
      - Sets the dataset path.
      - Initiates the recursive processing pipeline.

    Note: The JSON output is no longer generated.
    """
    # Set the root folder for your dataset (adjust as needed)
    dataset_root = "groove"  # adjust to the location where your dataset is stored

    # Set the output folder for bar arrays
    output_root = "bar_arrays"

    process_dataset(dataset_root, output_root)
    print("All MIDI files processed and bar arrays saved.")

if __name__ == "__main__":
    main()
