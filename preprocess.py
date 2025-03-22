"""
This module processes a MIDI file for drummer groove data. It extracts tempo, time signature,
note events, and specific control change events (hi-hat pedal) from the MIDI file, grouping these events into bars.
Enhanced functionality:
    • Adds a "normalized_onset_position_in_bar_by_beat" field as well as the "normalized_onset_index" field
      to each note event. The normalized onset value is computed by rounding the onset position (in beats)
      to the nearest value defined in a custom grid and saving the index of that value.
    • Groups notes into bars based on the normalized position: if the normalized value is 4.0 then the note is
      actually assigned to the next bar and its normalized onset is forced to 0.0.
"""

import json
import numpy as np
from typing import Tuple, Any, Dict, List
from mido import MidiFile, Message, bpm2tempo, tempo2bpm

# Create division grids
sixteenth_divs = np.linspace(0, 4, 32, endpoint=False)
eighth_triplet_divs = np.linspace(0, 4, 24, endpoint=False)
total_divs = set(sixteenth_divs).union(set(eighth_triplet_divs))
total_divs = sorted(list(total_divs))
total_divs.append(4.0)


def closest_value(target, values=tuple(total_divs)) -> Tuple[float, int]:
    """
    Returns a tuple (closest_value, index) where closest_value is the value in 'values'
    that is closest to the given 'target' and index is its index in the tuple 'values'.

    Parameters:
        target (float): The value to compare against.
        values (tuple, optional): Tuple of candidate values (default: total_divs).

    Returns:
        Tuple[float, int]: The closest value and its index.
    """
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
            - MidiFile: Object representation of the MIDI file (for access to ticks_per_beat).
            - List[Message]: The first track from the MIDI file.
    """
    midi = MidiFile(midi_file)
    if not midi.tracks:
        raise ValueError("The MIDI file does not contain any tracks.")
    return midi, midi.tracks[0]


def get_track_info(messages: List[Message]) -> Tuple[int, float, int, int, int, int]:
    """
    Extracts global tempo and time signature information from the track messages.
    Default values are used if specific meta messages are not found.

    Parameters:
        messages (List[Message]): A list of MIDI messages from which metadata is to be extracted.

    Returns:
        Tuple containing:
            - tempo (int): The microseconds per beat (default: 50000).
            - bpm (float): Beats per minute (default: 120).
            - numerator (int): Time signature numerator (default: 4).
            - denominator (int): Time signature denominator (default: 4).
            - clocks_per_click (int): MIDI clocks per metronome click (default: 24).
            - notated_32nd_notes_per_beat (int): Number of notated 32nd notes per beat (default: 8).
    """
    # Default meta event values:
    tempo = 50000  # microseconds per beat; default value in case no set_tempo is found
    bpm = 120      # default BPM
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
    Processes MIDI track messages to extract note events and hi-hat pedal control events,
    grouping them into bars and including note family information.

    Also calculates the note's onset position in the bar in beat units (rounded to 2 decimals)
    along with a normalized value based on custom rhythmic subdivisions.
    IMPORTANT CHANGE: Instead of grouping notes by the absolute position (using the modulo of ticks_per_bar),
    we check the normalized onset value. If that value equals 4.0 then the note is considered to start at 0.0
    of the next bar.

    Parameters:
        track (List[Message]): A list of MIDI messages (from a single track).
        ticks_per_beat (int): Ticks per beat as defined in the MIDI file.

    Returns:
        Dict[str, Any]: A dictionary keyed by bar number (e.g., 'bar_1'), each containing:
            - "notes": List of dictionaries for note_on events (nonzero velocity).
              Each note event includes:
                • "onset_position_in_bar_by_beat": Beat-level representation (rounded to 2 decimals).
                • "normalized_onset_position_in_bar_by_beat": The normalized beat value from our custom grid.
                • "normalized_onset_index": The index of the normalized value within the grid.
    """

    # Mapping of MIDI note numbers to their respective families.
    note_family_mapping = {
        36: 1, 38: 2, 40: 2, 37: 2, 48: 3, 50: 3,
        45: 4, 47: 4, 43: 5, 58: 5, 46: 6, 26: 6,
        42: 7, 22: 7, 44: 7, 49: 8, 55: 8, 57: 9,
        52: 9, 51: 10, 59: 10, 53: 10
    }

    # Retrieve global tempo and time signature values using the first few track messages.
    tempo, bpm, numerator, denominator, clocks_per_click, notated_32nd_notes_per_beat = get_track_info(track[:5])

    # Compute ticks per bar.
    ticks_per_bar = ticks_per_beat * (4 * numerator // denominator)

    # Dictionary to hold bar-wise event data.
    bars: Dict[str, Dict[str, List[Any]]] = {}

    abs_time = 0  # Running tally of absolute time in ticks.

    for msg in track:
        abs_time += msg.time  # msg.time is delta ticks from previous event

        # Process note_on events with nonzero velocity (zero velocity is considered note_off).
        if msg.type == 'note_on' and msg.velocity != 0:
            # The default bar based solely on abs_time:
            base_bar = abs_time // ticks_per_bar
            onset_in_bar = abs_time % ticks_per_bar
            onset_in_beats = onset_in_bar / ticks_per_beat

            # Compute normalized onset using our grid.
            normalized_onset, normalized_index = closest_value(onset_in_beats)

            # IMPORTANT: if the normalized value is 4.0, treat it as the beginning of the next bar.
            if normalized_onset == 4.0:
                normalized_onset = 0.0
                onset_in_beats = 0.0
                # Update normalized_index to the index corresponding to 0.0 in the grid
                normalized_index = total_divs.index(0.0)
                base_bar += 1

            # round values
            onset_in_beats = round(onset_in_beats, 2)
            normalized_onset = round(normalized_onset, 3)

            # Create the bar key as a 1-indexed number.
            bar_key = f"bar_{base_bar + 1}"

            family_index = note_family_mapping.get(msg.note, None)  # Get family index or None if not found

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


def main() -> None:
    """
    Main routine: processes an example MIDI file, extracts event data,
    and prints the result.
    """
    # Path to the MIDI file to be processed.
    midi_path = "groove/drummer1/session1/4_jazz-funk_116_beat_4-4.mid"

    try:
        midi_obj, track_obj = get_track_object(midi_path)
    except Exception as e:
        print(f"Error loading MIDI file: {e}")
        return

    # Process the first track using the ticks_per_beat information.
    processed_track = process_track(track_obj, midi_obj.ticks_per_beat)

    print('success')

    # Save to JSON for readability.
    with open('track_info.json', 'w') as json_file:
        json.dump(processed_track, json_file, indent=4)  # indent=4 is optional for pretty printing.


if __name__ == '__main__':
    main()
