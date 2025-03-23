import os
import threading
import time
from queue import Queue

import mido
import numpy as np
from mido import Message, open_output

TOTAL_NUM_FAMILIES = 10
TOTAL_TIME_LOCATIONS = 192

# Example drum mapping
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
    10: 51,  # Ride
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


############################
# Real-Time Bar Player
############################


def play_bar_realtime(
    bar_array: np.ndarray,
    outport: mido.ports.BaseOutput,
    bpm: float = 120.0,
    tpb: int = 480,
    note_length_ticks: int = 30,
):
    """
    Plays a single bar (drum notes) in real time. Blocks until the bar is finished.
    bar_array: shape [10, 48]
    """

    # Build note_on / note_off events
    grid = get_division_grid()
    events = []
    for col in range(TOTAL_TIME_LOCATIONS):
        onset_tick = int(round(grid[col] * tpb))
        for row in range(TOTAL_NUM_FAMILIES):
            cell_value = bar_array[row, col]

            velocity = int(max(cell_value * 127, 0))
            if velocity > 0:
                midi_note = family_to_midi.get(row + 1)
                if midi_note is None:
                    continue
                # note_on event
                events.append(
                    (onset_tick, Message("note_on", note=midi_note, velocity=velocity))
                )
                # note_off event after some ticks
                off_tick = onset_tick + note_length_ticks
                events.append(
                    (off_tick, Message("note_off", note=midi_note, velocity=0))
                )

    # Sort by absolute tick
    events.sort(key=lambda x: x[0])

    # Real-time scheduling
    start_time = time.monotonic()
    for abs_tick, msg in events:
        event_time_offset = ticks_to_seconds(abs_tick, bpm, tpb)
        scheduled_time = start_time + event_time_offset

        # Wait until we reach scheduled_time
        while True:
            now = time.monotonic()
            if now >= scheduled_time:
                break
            time.sleep(0.0005)  # short sleep to reduce busy-wait

        outport.send(msg)


def bar_player_loop(
    bar_queue: Queue,
    output_port: str = "IAC Driver Bus 1",
    bpm: float = 120.0,
    tpb: int = 480,
):
    """
    Continuously runs, taking bar_arrays from bar_queue and playing them.
    Blocks if no bars are in the queue.
    """
    print(f"Bar player started, opening MIDI port: {output_port}")
    try:
        with open_output(output_port) as outport:
            while True:
                # Get the next bar (BLOCK until available)
                bar_array = bar_queue.get()  # blocks if empty
                if bar_array is None:
                    # If we get None, it signals "stop"
                    print("Received None in bar_queue, stopping playback.")
                    break

                # Play the bar in real time
                play_bar_realtime(bar_array, outport=outport, bpm=bpm, tpb=tpb)

                # When finished, we'll loop and grab the next bar
    except Exception as e:
        print(f"Error in bar_player_loop: {e}")


############################
# Example: Generator
############################


def generate_fake_bar(npy_file="generated_output.npy") -> np.ndarray:
    with open(npy_file, "rb") as f:
        bar_array = np.load(f)
    return bar_array


def generator_loop(bar_queue: Queue, bars_to_generate=8):
    """
    Continuously generate bars (perhaps triggered by your model),
    then push them into the bar_queue for playback.
    bars_to_generate: how many bars we want to push in this example
    """
    print("Starting generator loop...")
    for i in range(bars_to_generate):
        # Could be your actual GAN inference
        bar_array = generate_fake_bar()
        # Put in the queue
        bar_queue.put(bar_array)
        print(f"Generated bar #{i+1}, queued for playback.")
        # Wait some time before next generation, or generate immediately
        time.sleep(1.0)

    # Once done, we can optionally push a None to signal no more bars
    bar_queue.put(None)
    print("Generator is done, pushed None to stop the player.")


############################
# Main Entry Point
############################


def main():
    # A thread-safe queue for bars
    bar_queue = Queue()

    # Start the bar_player_loop in a thread
    player_thread = threading.Thread(
        target=bar_player_loop,
        args=(bar_queue, "IAC Driver Bus 1", 50.0, 480),
        daemon=True,
    )
    player_thread.start()

    # Start the generator loop in the main thread (or another thread)
    # This simulates or calls your actual GAN inference code
    generator_loop(bar_queue, bars_to_generate=4)

    # Wait for the player to finish
    # (player finishes when it sees None in the queue)
    player_thread.join()
    print("All done!")


if __name__ == "__main__":
    main()
