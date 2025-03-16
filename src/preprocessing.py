from music21 import converter

# Load the MIDI file
midi_path = "dataset/drummer1/session1/4_jazz-funk_116_beat_4-4.mid"  # Replace with your actual MIDI file path
midi_score = converter.parse(midi_path)

# Show basic information about the parsed MIDI file
print(midi_score)

# Iterate through parts (MIDI tracks)
for part in midi_score.parts:
    print(f"Part: {part.partName}")  # Show the part name (if available)
    
    # Iterate through measures and notes
    for element in part.recurse():
        if element.isNote:
            print(f"Note: {element.name}, Pitch: {element.pitch}, Duration: {element.quarterLength}")
        elif element.isChord:
            print(f"Chord: {[n.name for n in element.notes]}, Duration: {element.quarterLength}")
