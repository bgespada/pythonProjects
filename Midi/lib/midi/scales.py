"""
Musical scale definitions for the sequencer.
Each scale is a list of semitone intervals from the root note.
"""

NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

# Root notes mapped to MIDI note numbers (octave 3, e.g. C3 = 48)
ROOT_NOTES: dict[str, int] = {
    "C": 48, "C#": 49, "D": 50, "D#": 51,
    "E": 52, "F": 53, "F#": 54, "G": 55,
    "G#": 56, "A": 57, "A#": 58, "B": 59,
}

# Scale families and their interval patterns (semitones from root)
SCALE_FAMILIES: dict[str, dict[str, list[int]]] = {
    "Diatonic": {
        "Major":           [0, 2, 4, 5, 7, 9, 11],
        "Natural Minor":   [0, 2, 3, 5, 7, 8, 10],
        "Harmonic Minor":  [0, 2, 3, 5, 7, 8, 11],
        "Melodic Minor":   [0, 2, 3, 5, 7, 9, 11],
    },
    "Pentatonic": {
        "Major Pentatonic": [0, 2, 4, 7, 9],
        "Minor Pentatonic": [0, 3, 5, 7, 10],
        "Blues":            [0, 3, 5, 6, 7, 10],
    },
    "Modes": {
        "Dorian":     [0, 2, 3, 5, 7, 9, 10],
        "Phrygian":   [0, 1, 3, 5, 7, 8, 10],
        "Lydian":     [0, 2, 4, 6, 7, 9, 11],
        "Mixolydian": [0, 2, 4, 5, 7, 9, 10],
        "Locrian":    [0, 1, 3, 5, 6, 8, 10],
    },
    "Exotic": {
        "Whole Tone":     [0, 2, 4, 6, 8, 10],
        "Diminished":     [0, 2, 3, 5, 6, 8, 9, 11],
        "Chromatic":      list(range(12)),
        "Hungarian Minor":[0, 2, 3, 6, 7, 8, 11],
    },
}


def note_name(midi_note: int) -> str:
    """Return human-readable note name for a MIDI note number (e.g. 60 → 'C4')."""
    octave = midi_note // 12 - 1
    name = NOTE_NAMES[midi_note % 12]
    return f"{name}{octave}"


def generate_notes(root_midi: int, intervals: list[int], octaves: int = 2) -> list[int]:
    """
    Generate a list of MIDI note numbers for the given scale.

    Args:
        root_midi: MIDI note number of the root (e.g. 48 for C3).
        intervals: Semitone intervals from root defining the scale.
        octaves: Number of octaves to generate (default 2).

    Returns:
        List of MIDI note numbers in ascending order.
    """
    notes = []
    for oct in range(octaves):
        for interval in intervals:
            note = root_midi + oct * 12 + interval
            if 0 <= note <= 127:
                notes.append(note)
    return notes
