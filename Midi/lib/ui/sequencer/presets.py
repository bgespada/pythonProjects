from typing import Any


PRESET_NAMES = [
    "Empty",
    "Up Arp",
    "Down Arp",
    "Pulse Root",
]


def build_preset(name: str, scale_notes: list[int], note_names: list[str], num_steps: int) -> dict[str, Any]:
    """
    Build a preset pattern dict for the current scale and step count.
    """
    notes_by_step: list[list[int]] = [[] for _ in range(num_steps)]
    step_lengths = [1] * num_steps
    step_velocities = [100] * num_steps

    if not scale_notes:
        return {
            "version": 1,
            "num_steps": num_steps,
            "scale_notes": [],
            "note_names": [],
            "step_lengths": step_lengths,
            "step_velocities": step_velocities,
            "notes_by_step": notes_by_step,
        }

    if name == "Up Arp":
        for step in range(num_steps):
            note = scale_notes[step % len(scale_notes)]
            notes_by_step[step] = [note]
    elif name == "Down Arp":
        rev = list(reversed(scale_notes))
        for step in range(num_steps):
            note = rev[step % len(rev)]
            notes_by_step[step] = [note]
    elif name == "Pulse Root":
        root = scale_notes[0]
        octave = root + 12 if root + 12 <= 127 else root
        for step in range(num_steps):
            if step % 4 == 0:
                notes_by_step[step] = [root]
                step_velocities[step] = 115
            elif step % 4 == 2:
                notes_by_step[step] = [octave]
                step_velocities[step] = 95
    # "Empty" or unknown keep defaults

    return {
        "version": 1,
        "num_steps": num_steps,
        "scale_notes": scale_notes,
        "note_names": note_names,
        "step_lengths": step_lengths,
        "step_velocities": step_velocities,
        "notes_by_step": notes_by_step,
    }
