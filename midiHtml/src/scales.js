/**
 * scales.js
 * Scale definitions and MIDI note generation.
 * Mirrors scales.py from the Python project.
 */

const NOTE_NAMES = ['C','C#','D','D#','E','F','F#','G','G#','A','A#','B'];

// Scale families and their interval patterns (semitones from root)
const SCALE_FAMILIES = {
  Diatonic: {
    Major:            [0,2,4,5,7,9,11],
    'Natural Minor':  [0,2,3,5,7,8,10],
    'Harmonic Minor': [0,2,3,5,7,8,11],
    'Melodic Minor':  [0,2,3,5,7,9,11],
    Dorian:           [0,2,3,5,7,9,10],
    Phrygian:         [0,1,3,5,7,8,10],
    Lydian:           [0,2,4,6,7,9,11],
    Mixolydian:       [0,2,4,5,7,9,10],
    Locrian:          [0,1,3,5,6,8,10],
  },
  Pentatonic: {
    'Major Pentatonic': [0,2,4,7,9],
    'Minor Pentatonic': [0,3,5,7,10],
    'Blues':            [0,3,5,6,7,10],
    'Hirajoshi':        [0,2,3,7,8],
    'Insen':            [0,1,5,7,10],
  },
  Modes: {
    Ionian:     [0,2,4,5,7,9,11],
    Aeolian:    [0,2,3,5,7,8,10],
    Lydian:     [0,2,4,6,7,9,11],
    Mixolydian: [0,2,4,5,7,9,10],
    Dorian:     [0,2,3,5,7,9,10],
    Phrygian:   [0,1,3,5,7,8,10],
    Locrian:    [0,1,3,5,6,8,10],
  },
  Exotic: {
    'Whole Tone':     [0,2,4,6,8,10],
    'Diminished':     [0,2,3,5,6,8,9,11],
    'Chromatic':      [0,1,2,3,4,5,6,7,8,9,10,11],
    'Hungarian Minor':[0,2,3,6,7,8,11],
    'Double Harmonic':[0,1,4,5,7,8,11],
    'Enigmatic':      [0,1,4,6,8,10,11],
  },
};

/**
 * Generate an array of note numbers for the given root + scale across multiple
 * octaves so we have enough notes for a piano roll.
 *
 * rootName: 'C','C#',... 
 * intervals: array of semitones from root
 * octaves: how many octaves to span (default 3, starting at octave 4)
 * startOctave: MIDI octave start (default 4, so C4=60)
 *
 * Returns Array of { midi, name } sorted ascending.
 */
function generateScaleNotes(rootName, intervals, octaves = 3, startOctave = 4) {
  const rootOffset = NOTE_NAMES.indexOf(rootName);
  if (rootOffset === -1) return [];
  const notes = [];
  for (let oct = startOctave; oct < startOctave + octaves; oct++) {
    for (const interval of intervals) {
      const midi = (oct + 1) * 12 + rootOffset + interval;
      if (midi >= 0 && midi <= 127) {
        const noteName = NOTE_NAMES[(rootOffset + interval) % 12];
        notes.push({ midi, name: `${noteName}${oct}` });
      }
    }
  }
  // Deduplicate and sort
  const seen = new Set();
  return notes
    .filter((n) => { if (seen.has(n.midi)) return false; seen.add(n.midi); return true; })
    .sort((a, b) => a.midi - b.midi);
}
