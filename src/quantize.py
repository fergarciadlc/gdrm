from music21 import converter, instrument

midi_path = "dataset/drummer1/session1/4_jazz-funk_116_beat_4-4.mid"
midi_score = converter.parse(midi_path)

sixteenth_duration = 0.25  # Duration of a sixteenth note

for part in midi_score.parts:
    instr = part.getInstrument()
    if isinstance(instr, instrument.UnpitchedPercussion):
        for note in part.flat.notes:
            note.offset = round(note.offset / sixteenth_duration) * sixteenth_duration
            note.quarterLength = round(note.quarterLength / sixteenth_duration) * sixteenth_duration

midi_score.write('midi', 'quantized_drums.mid')