"""
Musical scale definitions for the sequencer.
Each scale is a list of semitone intervals from the root note.
"""

NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
MIN_MIDI_NOTE = 0
MAX_MIDI_NOTE = 127

# Root notes mapped to MIDI note numbers in octave -2 (C-2 = 0)
ROOT_NOTES: dict[str, int] = {
    "C": 0, "C#": 1, "D": 2, "D#": 3,
    "E": 4, "F": 5, "F#": 6, "G": 7,
    "G#": 8, "A": 9, "A#": 10, "B": 11,
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
    """Return note name using the convention C-2 = MIDI note 0."""
    octave = midi_note // 12 - 2
    name = NOTE_NAMES[midi_note % 12]
    return f"{name}{octave}"


def generate_notes(
    root_midi: int,
    intervals: list[int],
    min_midi_note: int = MIN_MIDI_NOTE,
    max_midi_note: int = MAX_MIDI_NOTE,
) -> list[int]:
    """
    Generate scale notes across a MIDI-note range.

    Args:
        root_midi: MIDI note number for root pitch class anchor (0..11 recommended).
        intervals: Semitone intervals from root defining the scale.
        min_midi_note: Lowest MIDI note to include.
        max_midi_note: Highest MIDI note to include.

    Returns:
        List of MIDI note numbers in ascending order.
    """
    min_midi_note = max(MIN_MIDI_NOTE, min_midi_note)
    max_midi_note = min(MAX_MIDI_NOTE, max_midi_note)
    if min_midi_note > max_midi_note:
        return []

    normalized_intervals = {interval % 12 for interval in intervals}
    root_pitch_class = root_midi % 12

    return [
        note
        for note in range(min_midi_note, max_midi_note + 1)
        if ((note - root_pitch_class) % 12) in normalized_intervals
    ]
