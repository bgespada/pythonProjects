/**
 * ccParameters.js
 * All CC parameter definitions, mirroring the Python ccParameters.py.
 */

// Each parameter:
// { name, cc, type: 'slider'|'boolean'|'dropdown', min, max, default, options? }

const SOUND_ENGINE_ENGINES = [
  'OPERATOR_TYPE','FORMANT_OSCILLATOR_TYPE','HARMONIC_OSCILLATOR_TYPE',
  'MASS_SPRING_CHAIN_TYPE','STRING_VOICE_TYPE','MODAL_VOICE_TYPE',
  'KARPLUS_STRONG_TYPE','GRAIN_VOICE_TYPE','ANALOG_FILTER_TYPE',
  'VARIABLE_SHAPE_OSCILLATOR_TYPE','WAVETABLES_SAMPLER_TYPE',
  'BASIC_WAVETABLE_TYPE','FEEDBACK_FM_TYPE','WAVEFOLDER_TYPE',
  'RESONATOR_TYPE','PHASE_DISTORTION_TYPE','BITCRUSHER_TYPE',
  'CHORD_SYNTH_TYPE','DRONE_SYNTH_TYPE',
];

// Per-engine slot definitions (up to 8 slots mapped to CC 102-109)
const ENGINE_PARAMS = {
  OPERATOR_TYPE: [
    { name: 'Ratio',     type: 'slider',  min: 0, max: 127, default: 0 },
    { name: 'Index',     type: 'slider',  min: 0, max: 127, default: 0 },
    { name: 'Amplitude', type: 'slider',  min: 0, max: 127, default: 127 },
    { name: 'Waveform',  type: 'slider',  min: 0, max: 127, default: 0 },
  ],
  FORMANT_OSCILLATOR_TYPE: [
    { name: 'Formant Frequency', type: 'slider', min: 0, max: 127, default: 64 },
    { name: 'Phase Shift',       type: 'slider', min: 0, max: 127, default: 0 },
  ],
  HARMONIC_OSCILLATOR_TYPE: [
    { name: 'First Harmonic', type: 'slider', min: 0, max: 127, default: 0 },
    { name: 'Amplitude',      type: 'slider', min: 0, max: 127, default: 127 },
    { name: 'Single Amp',     type: 'slider', min: 0, max: 127, default: 0 },
  ],
  MASS_SPRING_CHAIN_TYPE: [
    { name: 'Sustain',    type: 'slider',  min: 0, max: 127, default: 64 },
    { name: 'Brightness', type: 'slider',  min: 0, max: 127, default: 64 },
    { name: 'Damping',    type: 'slider',  min: 0, max: 127, default: 64 },
  ],
  STRING_VOICE_TYPE: [
    { name: 'Sustain',    type: 'slider',  min: 0, max: 127, default: 64 },
    { name: 'Accent',     type: 'slider',  min: 0, max: 127, default: 0 },
    { name: 'Structure',  type: 'slider',  min: 0, max: 127, default: 64 },
    { name: 'Brightness', type: 'slider',  min: 0, max: 127, default: 64 },
    { name: 'Damping',    type: 'slider',  min: 0, max: 127, default: 64 },
  ],
  MODAL_VOICE_TYPE: [
    { name: 'Sustain',    type: 'slider',  min: 0, max: 127, default: 64 },
    { name: 'Accent',     type: 'slider',  min: 0, max: 127, default: 0 },
    { name: 'Structure',  type: 'slider',  min: 0, max: 127, default: 64 },
    { name: 'Brightness', type: 'slider',  min: 0, max: 127, default: 64 },
    { name: 'Damping',    type: 'slider',  min: 0, max: 127, default: 64 },
  ],
  KARPLUS_STRONG_TYPE: [
    { name: 'Damping',    type: 'slider',  min: 0, max: 127, default: 64 },
    { name: 'Brightness', type: 'slider',  min: 0, max: 127, default: 64 },
  ],
  GRAIN_VOICE_TYPE: [
    { name: 'Position', type: 'slider', min: 0, max: 127, default: 0 },
    { name: 'Size',     type: 'slider', min: 0, max: 127, default: 64 },
    { name: 'Pitch',    type: 'slider', min: 0, max: 127, default: 64 },
    { name: 'Density',  type: 'slider', min: 0, max: 127, default: 64 },
    { name: 'Texture',  type: 'slider', min: 0, max: 127, default: 64 },
  ],
  ANALOG_FILTER_TYPE: [
    { name: 'Cutoff',    type: 'slider', min: 0, max: 127, default: 127 },
    { name: 'Resonance', type: 'slider', min: 0, max: 127, default: 0 },
  ],
  VARIABLE_SHAPE_OSCILLATOR_TYPE: [
    { name: 'PW',             type: 'slider',  min: 0, max: 127, default: 64 },
    { name: 'Waveshape',      type: 'slider',  min: 0, max: 127, default: 0 },
    { name: 'Sync Frequency', type: 'slider',  min: 0, max: 127, default: 0 },
    { name: 'Sync',           type: 'boolean', min: 0, max: 127, default: 0 },
  ],
  WAVETABLES_SAMPLER_TYPE: [
    { name: 'Waveform',      type: 'slider', min: 0, max: 127, default: 0 },
    { name: 'Morph Factor',  type: 'slider', min: 0, max: 127, default: 0 },
    { name: 'LFO Frequency', type: 'slider', min: 0, max: 127, default: 0 },
    { name: 'LFO Amount',    type: 'slider', min: 0, max: 127, default: 0 },
  ],
  BASIC_WAVETABLE_TYPE: [
    { name: 'Waveform', type: 'slider', min: 0, max: 127, default: 0 },
    { name: 'Morph',    type: 'slider', min: 0, max: 127, default: 0 },
  ],
  FEEDBACK_FM_TYPE: [
    { name: 'Ratio',    type: 'slider', min: 0, max: 127, default: 0 },
    { name: 'Feedback', type: 'slider', min: 0, max: 127, default: 0 },
    { name: 'Index',    type: 'slider', min: 0, max: 127, default: 0 },
  ],
  WAVEFOLDER_TYPE: [
    { name: 'Folds',       type: 'slider', min: 0, max: 127, default: 0 },
    { name: 'Symmetry',    type: 'slider', min: 0, max: 127, default: 64 },
  ],
  RESONATOR_TYPE: [
    { name: 'Frequency',  type: 'slider', min: 0, max: 127, default: 64 },
    { name: 'Decay',      type: 'slider', min: 0, max: 127, default: 64 },
    { name: 'Structure',  type: 'slider', min: 0, max: 127, default: 64 },
  ],
  PHASE_DISTORTION_TYPE: [
    { name: 'Distortion', type: 'slider', min: 0, max: 127, default: 0 },
    { name: 'Waveshape',  type: 'slider', min: 0, max: 127, default: 0 },
  ],
  BITCRUSHER_TYPE: [
    { name: 'Bit Depth',    type: 'slider', min: 0, max: 127, default: 127 },
    { name: 'Sample Rate',  type: 'slider', min: 0, max: 127, default: 127 },
  ],
  CHORD_SYNTH_TYPE: [
    { name: 'Chord Type', type: 'slider', min: 0, max: 127, default: 0 },
    { name: 'Spread',     type: 'slider', min: 0, max: 127, default: 0 },
  ],
  DRONE_SYNTH_TYPE: [
    { name: 'Interval', type: 'slider', min: 0, max: 127, default: 0 },
    { name: 'Detune',   type: 'slider', min: 0, max: 127, default: 0 },
  ],
};

function getEngineParams(engineName) {
  const params = (ENGINE_PARAMS[engineName] || []).slice(0, 8);
  return params.map((p, i) => ({ ...p, cc: 102 + i }));
}

// Fixed categories
const PARAMETER_CATEGORIES = [
  {
    name: 'Sound Engine',
    params: [
      { name: 'Engine', cc: 0, type: 'dropdown', min: 0, max: 18, default: 0,
        options: SOUND_ENGINE_ENGINES },
    ],
  },
  {
    name: 'Filter',
    params: [
      { name: 'Cutoff Frequency', cc: 74, type: 'slider', min: 0, max: 127, default: 64 },
      { name: 'Resonance',        cc: 71, type: 'slider', min: 0, max: 127, default: 0 },
      { name: 'Drive',            cc: 75, type: 'slider', min: 0, max: 127, default: 0 },
    ],
  },
  {
    name: 'Envelope',
    params: [
      { name: 'Attack',  cc: 12, type: 'slider', min: 0, max: 127, default: 0 },
      { name: 'Decay',   cc: 13, type: 'slider', min: 0, max: 127, default: 64 },
      { name: 'Sustain', cc: 14, type: 'slider', min: 0, max: 127, default: 100 },
      { name: 'Release', cc: 15, type: 'slider', min: 0, max: 127, default: 32 },
    ],
  },
  {
    name: 'Ping Pong Delay',
    params: [
      { name: 'Gain',       cc: 16, type: 'slider', min: 0, max: 127, default: 0 },
      { name: 'Time Left',  cc: 17, type: 'slider', min: 0, max: 127, default: 0 },
      { name: 'Time Right', cc: 18, type: 'slider', min: 0, max: 127, default: 0 },
      { name: 'Feedback',   cc: 19, type: 'slider', min: 0, max: 127, default: 0 },
      { name: 'HP Filter',  cc: 20, type: 'slider', min: 0, max: 127, default: 0 },
    ],
  },
  {
    name: 'Stereo Freeverb',
    params: [
      { name: 'Room Size', cc: 21, type: 'slider', min: 0, max: 127, default: 0 },
      { name: 'Damping',   cc: 22, type: 'slider', min: 0, max: 127, default: 0 },
      { name: 'Wet',       cc: 23, type: 'slider', min: 0, max: 127, default: 0 },
      { name: 'Dry',       cc: 24, type: 'slider', min: 0, max: 127, default: 0 },
      { name: 'Pre Delay', cc: 25, type: 'slider', min: 0, max: 127, default: 0 },
      { name: 'HP Filter', cc: 26, type: 'slider', min: 0, max: 127, default: 0 },
      { name: 'Resonance', cc: 27, type: 'slider', min: 0, max: 127, default: 0 },
      { name: 'Gain',      cc: 28, type: 'slider', min: 0, max: 127, default: 0 },
    ],
  },
  {
    name: 'Oscillator',
    params: [
      { name: 'Wave Shape', cc: 70, type: 'slider', min: 0, max: 127, default: 0 },
      { name: 'Detune',     cc: 94, type: 'slider', min: 0, max: 127, default: 64 },
    ],
  },
];
