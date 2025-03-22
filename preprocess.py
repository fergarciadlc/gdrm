"""
This module processes a MIDI file for drummer groove data. It extracts tempo,
time signature, note events, and specific control change events (hi-hat pedal) from the MIDI file,
grouping these events into bars.

Enhanced functionality:
    • Adds a "normalized_onset_position_in_bar_by_beat" field as well as the "normalized_onset_index" field
      to each note event. The normalized onset value is computed by rounding the onset position (in beats)
      to the nearest value defined in a custom grid and saving the index of that value.
    • Groups notes into bars based on the normalized position: if the normalized value is 4.0 then the note is
      actually assigned to the next bar and its normalized onset is forced to 0.0.
    • Converts each bar’s note events into a numpy array with dimensions 10 (families) x 48 (grid locations).
      The resulting arrays are saved to a folder called "bar_arrays" as .npy files.
"""

TOTAL_NUM_FAMILIES = 10
TOTAL_TIME_LOCATIONS = 192

import json
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

def create_bar_arrays(bars: Dict[str, Any]) -> None:
    """
    Converts each bar's note events into a numpy array and saves it to a .npy file.

    The array has dimensions:
        total_num_families x possible_time_locations
        In this example, the dimensions are 10 x 48.
    Each note event is placed using its family index (converted to 0-indexed row)
    and its normalized onset index as its column. If multiple notes are found at the same
    position, the velocity values are summed.

    Parameters:
        bars (Dict[str, Any]): Dictionary mapping bar keys to their note events.
    """
    total_num_families = TOTAL_NUM_FAMILIES
    total_time_locations = TOTAL_TIME_LOCATIONS

    # Create folder for numpy arrays if it doesn't exist
    output_folder = "bar_arrays"
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for bar_key, bar_data in bars.items():
        # Initialize an empty array for the bar.
        bar_array = np.full((total_num_families, total_time_locations), -1.0 , dtype=np.int32)

        for note in bar_data["notes"]:
            family_index = note.get("family_index")
            normalized_index = note.get("normalized_onset_index")
            velocity = note.get("velocity")
            # Proceed only if family_index is defined.
            if family_index is not None:
                row = family_index - 1  # convert to 0-indexed
                col = normalized_index        # provided index from grid calculation
                # Accumulate velocity if multiple notes fall into the same cell.
                bar_array[row, col] += velocity

        # Save the bar array as a .npy file.
        output_path = os.path.join(output_folder, f"{bar_key}.npy")
        np.save(output_path, bar_array)

def main() -> None:
    """
    Main routine that processes a MIDI file, writes the extracted track data to a JSON file,
    and saves numpy array representations for each bar in the 'bar_arrays' folder.
    """
    midi_path = "groove/drummer1/session1/4_jazz-funk_116_beat_4-4.mid"

    try:
        midi_obj, track_obj = get_track_object(midi_path)
    except Exception as e:
        print(f"Error loading MIDI file: {e}")
        return

    processed_track = process_track(track_obj, midi_obj.ticks_per_beat)
    print("Processing success!")

    # Save track info to a JSON file.
    with open("track_info.json", "w") as json_file:
        json.dump(processed_track, json_file, indent=4)

    # Create numpy array representations for each bar.
    create_bar_arrays(processed_track)
    print("Bar arrays saved in folder 'bar_arrays'.")

if __name__ == "__main__":
    main()
