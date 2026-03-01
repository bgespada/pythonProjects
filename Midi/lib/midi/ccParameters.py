"""
MIDI CC parameter definitions for the Eurorack MIDI-to-CV module.
Each category groups related parameters with their CC number and value range.
"""

PARAMETER_CATEGORIES: dict[str, list[dict]] = {
    "Filter": [
        {"name": "Cutoff Frequency",    "cc": 74, "min": 0, "max": 127, "default": 64},
        {"name": "Resonance",           "cc": 71, "min": 0, "max": 127, "default": 0},
        {"name": "Drive",               "cc": 75, "min": 0, "max": 127, "default": 0},
    ],
    "Envelope": [
        {"name": "Attack",              "cc": 12, "min": 0, "max": 127, "default": 0},
        {"name": "Decay",               "cc": 13, "min": 0, "max": 127, "default": 64},
        {"name": "Sustain",             "cc": 14, "min": 0, "max": 127, "default": 100},
        {"name": "Release",             "cc": 15, "min": 0, "max": 127, "default": 32},
    ],
    "LFO": [
        {"name": "Rate",                "cc": 76, "min": 0, "max": 127, "default": 64},
        {"name": "Depth",               "cc": 1,  "min": 0, "max": 127, "default": 0},
    ],
    "Oscillator": [
        {"name": "Detune",              "cc": 94, "min": 0, "max": 127, "default": 64},
        {"name": "Wave Shape",          "cc": 70, "min": 0, "max": 127, "default": 0},
    ],
}
