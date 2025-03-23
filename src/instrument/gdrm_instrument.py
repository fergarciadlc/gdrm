import os
import time
import numpy as np
import mido
from mido import Message, open_output

TOTAL_NUM_FAMILIES = 10
TOTAL_TIME_LOCATIONS = 192

# Example mapping of "drum families" to representative MIDI note numbers:
family_to_midi = {
    1: 36,  # Kick
    2: 38,  # Snare
    3: 48,
    4: 45,
    5: 43,
    6: 46,
    7: 42,  # Hi-Hat closed
    8: 49,  # Crash
    9: 57,
    10: 51  # Ride
}

def get_division_grid() -> list:
    """
    Recreates the custom division grid used in preprocessing:
    - 16th-note subdivisions (128 from 0..4) plus
    - 8th-note triplet subdivisions (96 from 0..4),
    then appended with 4.0 (the end of the bar).
    """
    sixteenth_divs = np.linspace(0, 4, 128, endpoint=False)
    eighth_triplet_divs = np.linspace(0, 4, 96, endpoint=False)
    grid = sorted(set(sixteenth_divs).union(set(eighth_triplet_divs)))
    grid.append(4.0)
    return grid

def ticks_to_seconds(ticks, bpm=120.0, tpb=480):
    """
    Convert 'ticks' to real seconds based on the BPM and ticks_per_beat (tpb).
    seconds_per_beat = 60 / bpm
    seconds_per_tick = (60 / bpm) / tpb
    """
    return ticks * (60.0 / (bpm * tpb))

def create_realtime_midi_from_array(
    npy_file: str,
    output_port: str = "IAC Driver Bus 1",
    bpm: float = 120.0,
    ticks_per_beat: int = 480,
    note_length_ticks: int = 30
):
    """
    1) Loads the numpy array (shape: [10, 48]).
    2) Builds a list of events, each with an 'absolute tick' and a Mido message.
       - We make both note_on and note_off events. note_off = note_on_tick + note_length_ticks
    3) Sorts all events by their time in ticks, so simultaneous events get played together.
    4) Uses monotonic time to schedule each event in real time, so they play at the correct moments.
    5) Sends all MIDI messages to the specified output_port in real time.
    """

    # --- Load the bar array (drum families x time-grid) ---
    try:
        bar_array = np.load(npy_file)  # e.g., shape [10, 48]
    except Exception as e:
        print(f"Error loading npy file: {e}")
        return

    grid = get_division_grid()
    if len(grid) < TOTAL_TIME_LOCATIONS:
        print("Warning: grid length is less than total time locations. Check grid generation!")

    # --- Build note_on + note_off events ---
    # We'll store them as a list of tuples: (abs_tick, mido.Message)
    events = []

    # 1) Create note_on events
    for col in range(TOTAL_TIME_LOCATIONS):
        onset_tick = int(round(grid[col] * ticks_per_beat))
        for row in range(TOTAL_NUM_FAMILIES):
            cell_value = bar_array[row, col]
            if cell_value > -1:  # indicates a note event
                velocity = int(cell_value + 1)  # original velocity
                midi_note = family_to_midi.get(row + 1)
                if midi_note is None:
                    continue

                # Note-On event
                msg_on = Message('note_on', note=midi_note, velocity=velocity, time=0)
                events.append((onset_tick, msg_on))

                # 2) Create note_off event a short time later
                off_tick = onset_tick + note_length_ticks
                msg_off = Message('note_off', note=midi_note, velocity=0, time=0)
                events.append((off_tick, msg_off))

    # --- Sort events by absolute tick ---
    events.sort(key=lambda x: x[0])

    # --- Open MIDI port ---
    print(f"Opening MIDI output port: {output_port}")
    try:
        outport = open_output(output_port)
    except Exception as e:
        print(f"Could not open MIDI port '{output_port}': {e}")
        return

    print(f"Starting real-time playback at {bpm} BPM...")
    start_time = time.monotonic()  # reference moment

    # --- Schedule events in real time ---
    for abs_tick, msg in events:
        # Convert ticks to seconds offset from the start of the bar
        event_time_offset = ticks_to_seconds(abs_tick, bpm=bpm, tpb=ticks_per_beat)
        scheduled_time = start_time + event_time_offset

        # Busy-wait or small-sleep until we reach the event's scheduled time
        while True:
            now = time.monotonic()
            if now >= scheduled_time:
                break
            # Sleep a bit to lighten CPU load; pick a small number for smooth timing
            time.sleep(0.0005)

        # Send the message
        outport.send(msg)

    outport.close()
    print("Playback completed.")

def main():
    # Example usage
    npy_file = os.path.join("bar_arrays", "1_funk_80_beat_4-4_bar_4.npy")
    create_realtime_midi_from_array(
        npy_file=npy_file,
        output_port="IAC Driver Bus 1",
        bpm=80.0,
        ticks_per_beat=480,
        note_length_ticks=30
    )

if __name__ == "__main__":
    main()