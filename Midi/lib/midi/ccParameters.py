"""
MIDI CC parameter definitions for the Eurorack MIDI-to-CV module.

This module now contains:
- Generic category definitions used by the current controls UI.
- Sound-engine metadata for the upcoming dedicated sound-engine UI:
  - dropdown selection of engine type
  - per-engine parameter schemas (slider / boolean)
"""

SOUND_ENGINES: list[str] = [
    "CLOCKED_NOISE_TYPE",
    "DRIP_TYPE",
    "DUST_TYPE",
    "OPERATOR_TYPE",
    "FORMANT_OSCILLATOR_TYPE",
    "FRACTAL_RANDOM_GENERATOR_TYPE",
    "GRAINLET_OSCILLATOR_TYPE",
    "HARMONIC_OSCILLATOR_TYPE",
    "MODAL_VOICE_TYPE",
    "OSCILLATOR_BANK_TYPE",
    "PHASOR_TYPE",
    "STRING_VOICE_TYPE",
    "VARIABLE_SAW_OSCILLATOR_TYPE",
    "VARIABLE_SHAPE_OSCILLATOR_TYPE",
    "VOSIM_OSCILLATOR_TYPE",
    "WHITE_NOISE_TYPE",
    "ZOSCILLATOR_TYPE",
    "KARPLUS_STRING_TYPE",
    "WAVETABLES_SAMPLER_TYPE",
]


SOUND_ENGINE_SELECTION_PARAMETER: dict = {
    "name": "Engine",
    "cc": 0,
    "min": 0,
    "max": len(SOUND_ENGINES) - 1,
    "default": 0,
    "widget": "dropdown",
    "options": SOUND_ENGINES,
}


SOUND_ENGINE_PARAM_SLOT_CCS: list[int] = [102, 103, 104, 105, 106, 107, 108, 109]


def _slider(name: str, default: int = 64) -> dict:
    return {
        "name": name,
        "widget": "slider",
        "min": 0,
        "max": 127,
        "default": default,
    }


def _boolean(name: str, default: int = 0) -> dict:
    return {
        "name": name,
        "widget": "boolean",
        "off_value": 0,
        "on_value": 127,
        "default": default,
    }


_SOUND_ENGINE_PARAMETERS_NO_CC: dict[str, list[dict]] = {
    "CLOCKED_NOISE_TYPE": [],
    "DRIP_TYPE": [],
    "DUST_TYPE": [
        _slider("Density"),
    ],
    "OPERATOR_TYPE": [
        _slider("Ratio"),
        _slider("Index"),
        _slider("Amplitude", default=100),
        _slider("Waveform", default=0),
    ],
    "FORMANT_OSCILLATOR_TYPE": [
        _slider("Formant frequency"),
        _slider("Phase shift"),
    ],
    "FRACTAL_RANDOM_GENERATOR_TYPE": [
        _slider("Color"),
    ],
    "GRAINLET_OSCILLATOR_TYPE": [
        _slider("Formant Frequency"),
        _slider("Shape"),
        _slider("Bleed"),
    ],
    "HARMONIC_OSCILLATOR_TYPE": [
        _slider("First Harmonic"),
        _slider("Amplitude", default=100),
        _slider("Single Amp", default=100),
    ],
    "MODAL_VOICE_TYPE": [
        _slider("Sustain"),
        _slider("Accent"),
        _slider("Structure"),
        _slider("Brighness"),
        _slider("Damping"),
    ],
    "OSCILLATOR_BANK_TYPE": [
        _slider("Amplitudes"),
        _slider("Single Amp", default=100),
        _slider("Gain", default=100),
    ],
    "PHASOR_TYPE": [],
    "STRING_VOICE_TYPE": [
        _slider("Sustain"),
        _slider("Accent"),
        _slider("Structure"),
        _slider("Brightness"),
        _slider("Damping"),
    ],
    "VARIABLE_SAW_OSCILLATOR_TYPE": [
        _slider("PW"),
        _slider("Waveshape"),
    ],
    "VARIABLE_SHAPE_OSCILLATOR_TYPE": [
        _slider("PW"),
        _slider("Waveshape"),
        _slider("Sync Frequency"),
        _boolean("Sync", default=0),
    ],
    "VOSIM_OSCILLATOR_TYPE": [
        _slider("Formant Frequency 1"),
        _slider("Formant Frequency 2"),
        _slider("Shape"),
    ],
    "WHITE_NOISE_TYPE": [],
    "ZOSCILLATOR_TYPE": [
        _slider("Formant Frequency"),
        _slider("Shape"),
        _slider("Mode", default=0),
    ],
    "KARPLUS_STRING_TYPE": [
        _slider("Non Linearity"),
        _slider("Brightness"),
        _slider("Damping"),
    ],
    "WAVETABLES_SAMPLER_TYPE": [
        _slider("Waveform", default=0),
        _slider("Morph Factor"),
        _slider("LFO Frequency"),
        _slider("LFO Amount"),
    ],
}


def _attach_engine_param_ccs(params: list[dict]) -> list[dict]:
    with_cc: list[dict] = []
    for index, param in enumerate(params):
        if index < len(SOUND_ENGINE_PARAM_SLOT_CCS):
            cc_value = SOUND_ENGINE_PARAM_SLOT_CCS[index]
        else:
            cc_value = SOUND_ENGINE_PARAM_SLOT_CCS[-1]
        enriched = dict(param)
        enriched["cc"] = cc_value
        with_cc.append(enriched)
    return with_cc


SOUND_ENGINE_PARAMETERS: dict[str, list[dict]] = {
    engine: _attach_engine_param_ccs(params)
    for engine, params in _SOUND_ENGINE_PARAMETERS_NO_CC.items()
}


PARAMETER_CATEGORIES: dict[str, list[dict]] = {
    "Sound Engine": [SOUND_ENGINE_SELECTION_PARAMETER],
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
    "Ping PongDelay": [
        {"name": "Time Left",           "cc": 17, "min": 0, "max": 127, "default": 0},
        {"name": "Time Right",          "cc": 18, "min": 0, "max": 127, "default": 0},
        {"name": "Feedback",            "cc": 19, "min": 0, "max": 127, "default": 0},
        {"name": "HP Filter",           "cc": 20, "min": 0, "max": 127, "default": 0},
        {"name": "Gain",                "cc": 16, "min": 0, "max": 127, "default": 0},
    ],
    "Stereo Freeverb": [
        {"name": "Room Size",           "cc": 21, "min": 0, "max": 127, "default": 0},
        {"name": "Damping",             "cc": 22, "min": 0, "max": 127, "default": 0},
        {"name": "Pre Delay",           "cc": 25, "min": 0, "max": 127, "default": 0},
        {"name": "HP Filter",           "cc": 26, "min": 0, "max": 127, "default": 0},
        {"name": "Resonance",           "cc": 27, "min": 0, "max": 127, "default": 0},
        {"name": "Wet",                 "cc": 23, "min": 0, "max": 127, "default": 0},
        {"name": "Dry",                 "cc": 24, "min": 0, "max": 127, "default": 0},
        {"name": "Gain",                "cc": 28, "min": 0, "max": 127, "default": 0},
    ],
    "Oscillator": [
        {"name": "Detune",              "cc": 94, "min": 0, "max": 127, "default": 64},
        {"name": "Wave Shape",          "cc": 70, "min": 0, "max": 127, "default": 0},
    ],
}


def get_sound_engine_parameters(engine_name: str) -> list[dict]:
    """Return parameter schema for a given sound engine name."""
    return list(SOUND_ENGINE_PARAMETERS.get(engine_name, []))
