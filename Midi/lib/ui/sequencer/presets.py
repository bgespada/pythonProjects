from typing import Any


UTILITY_PRESET_NAMES = [
    "Empty",
    "Up Arp",
    "Down Arp",
    "Pulse Root",
]

MUSICAL_PRESET_NAMES = [
    "Ride of the Valkyries",
    "Imperial March (Inspired)",
]

PRESET_NAMES = UTILITY_PRESET_NAMES + MUSICAL_PRESET_NAMES


MUSICAL_PRESET_SCALES: dict[str, dict[str, str]] = {
    "Ride of the Valkyries": {
        "root": "B",
        "family": "Diatonic",
        "scale": "Natural Minor",
    },
    "Imperial March (Inspired)": {
        "root": "G",
        "family": "Diatonic",
        "scale": "Natural Minor",
    },
}

MUSICAL_PRESET_STEPS: dict[str, int] = {
    "Ride of the Valkyries": 32,
    "Imperial March (Inspired)": 24,
}


def _nearest_scale_note(scale_notes: list[int], target: int) -> int:
    if not scale_notes:
        return max(0, min(127, target))
    return min(scale_notes, key=lambda note: abs(note - target))


def _anchor_note(scale_notes: list[int], preferred: int = 60) -> int:
    if not scale_notes:
        return preferred
    return _nearest_scale_note(scale_notes, preferred)


def build_preset(name: str, scale_notes: list[int], note_names: list[str], num_steps: int) -> dict[str, Any]:
    """
    Build a preset pattern dict for the current scale and step count.
    """
    preset_steps = MUSICAL_PRESET_STEPS.get(name, num_steps)

    notes_by_step: list[list[int]] = [[] for _ in range(preset_steps)]
    step_lengths = [1] * preset_steps
    step_velocities = [100] * preset_steps
    step_gates = [64] * preset_steps

    if not scale_notes:
        return {
            "version": 1,
            "num_steps": preset_steps,
            "scale_notes": [],
            "note_names": [],
            "step_lengths": step_lengths,
            "step_velocities": step_velocities,
            "step_gates": step_gates,
            "notes_by_step": notes_by_step,
        }

    if name == "Up Arp":
        for step in range(preset_steps):
            note = scale_notes[step % len(scale_notes)]
            notes_by_step[step] = [note]
    elif name == "Down Arp":
        rev = list(reversed(scale_notes))
        for step in range(preset_steps):
            note = rev[step % len(rev)]
            notes_by_step[step] = [note]
    elif name == "Pulse Root":
        root = scale_notes[0]
        octave = root + 12 if root + 12 <= 127 else root
        for step in range(preset_steps):
            if step % 4 == 0:
                notes_by_step[step] = [root]
                step_velocities[step] = 115
            elif step % 4 == 2:
                notes_by_step[step] = [octave]
                step_velocities[step] = 95
    elif name == "Ride of the Valkyries":
        anchor = _anchor_note(scale_notes, preferred=62)
        phrase_intervals = [0, 7, 12, 7, 0, 7, 12, 14]
        for step in range(preset_steps):
            if step % 2 == 1:
                continue
            phrase_idx = (step // 2) % len(phrase_intervals)
            target = anchor + phrase_intervals[phrase_idx]
            note = _nearest_scale_note(scale_notes, target)
            notes_by_step[step] = [note]
            step_velocities[step] = 112 if phrase_idx in {0, 2} else 98
            step_gates[step] = 84
            step_lengths[step] = 2 if phrase_idx in {2, 7} else 1
    elif name == "Imperial March (Inspired)":
        anchor = _anchor_note(scale_notes, preferred=55)
        motif_intervals = [0, 0, 0, -3, 7, 0, -3, 7]
        accent_steps = {0, 4, 8, 12, 16, 20}
        for step in range(preset_steps):
            if step % 2 == 1:
                continue
            motif_idx = (step // 2) % len(motif_intervals)
            target = anchor + motif_intervals[motif_idx]
            note = _nearest_scale_note(scale_notes, target)
            notes_by_step[step] = [note]
            step_velocities[step] = 120 if step in accent_steps else 102
            step_gates[step] = 72
            step_lengths[step] = 2 if motif_idx in {2, 4} else 1
    # "Empty" or unknown keep defaults

    return {
        "version": 1,
        "num_steps": preset_steps,
        "scale_notes": scale_notes,
        "note_names": note_names,
        "step_lengths": step_lengths,
        "step_velocities": step_velocities,
        "step_gates": step_gates,
        "notes_by_step": notes_by_step,
    }
