/**
 * sequencer.js
 * Piano roll grid + sequencer engine.
 * Mirrors PianoRollUi, SequencerEngine, PatternStorage, ScaleTreeUi, presets.py.
 */

// scales.js loaded via <script> tag

// ============================================================
// Presets
// ============================================================

const MUSICAL_PRESETS = {
  valkyries: {
    name: 'Ride of the Valkyries',
    // root C, Major scale, 32 steps — note indices into scale
    rootName: 'C',
    scaleName: 'Major',
    family: 'Diatonic',
    numSteps: 32,
    noteIndices: [4,4,6,4,6,4,2,4,4,6,4,6,4,2,7,7,9,7,9,7,5,7,7,9,7,9,7,5,2,4,5,7],
    velocities: Array(32).fill(100),
    lengths:    Array(32).fill(1),
    gates:      Array(32).fill(64),
  },
  imperial: {
    name: 'Imperial March (Inspired)',
    rootName: 'C',
    scaleName: 'Natural Minor',
    family: 'Diatonic',
    numSteps: 24,
    noteIndices: [0,0,0,5,2,0,5,2,0,7,7,7,8,2,0,5,2,0,5,2,0,7,7,7],
    velocities: Array(24).fill(100),
    lengths:    Array(24).fill(1),
    gates:      Array(24).fill(64),
  },
};

// ============================================================
// PianoRoll  (one canvas-based piano roll instance)
// ============================================================

const CELL_W = 36;
const CELL_H = 18;
const HEADER_H = 20;
const PARAM_ROW_H = 28;
const PARAM_ROWS = 3; // LEN, VEL, GATE
const PARAM_LABELS = ['LEN', 'VEL', 'GATE'];
const LEN_VALUES = ['1/16', '1/8', '1/4'];

class PianoRoll {
  constructor(container, numSteps, scaleNotes) {
    this._container = container;
    this._numSteps = numSteps;
    this._scaleNotes = scaleNotes; // [{midi, name}, ...]
    this._numRows = scaleNotes.length;

    // Step data — activeNotes[step] = Set of rowIndex
    this._activeNotes = Array.from({ length: numSteps }, () => new Set());
    this._stepLengths   = Array(numSteps).fill(0);   // index into LEN_VALUES
    this._stepVelocities= Array(numSteps).fill(100);
    this._stepGates     = Array(numSteps).fill(64);

    this._currentStep = -1;   // highlighted during playback

    this._build(); // _attachEvents() is called inside _build()
  }

  // ----------------------------------------------------------------
  // Build DOM
  // ----------------------------------------------------------------

  _build() {
    this._container.innerHTML = '';

    const totalW = this._numSteps * CELL_W;
    const gridH  = this._numRows * CELL_H;
    const paramH = PARAM_ROWS * PARAM_ROW_H;

    // Header canvas
    this._headerEl = this._makeWrapper('pr-header-wrap');
    this._headerCanvas = this._makeCanvas(totalW, HEADER_H);
    this._headerEl.appendChild(this._headerCanvas);
    this._container.appendChild(this._headerEl);

    // Grid canvas
    this._gridEl = this._makeWrapper('pr-grid-wrap');
    this._gridCanvas = this._makeCanvas(totalW, gridH);
    this._gridEl.appendChild(this._gridCanvas);
    this._container.appendChild(this._gridEl);

    // Param canvas
    this._paramEl = this._makeWrapper('pr-param-wrap');
    this._paramCanvas = this._makeCanvas(totalW, paramH);
    this._paramEl.appendChild(this._paramCanvas);
    this._container.appendChild(this._paramEl);

    // Sync horizontal scroll
    const syncScroll = (src, dsts) => {
      src.addEventListener('scroll', () => {
        dsts.forEach((d) => { d.scrollLeft = src.scrollLeft; });
      });
    };
    syncScroll(this._paramEl, [this._headerEl, this._gridEl]);
    syncScroll(this._gridEl,  [this._headerEl, this._paramEl]);

    this._drawAll();
    this._attachEvents();
  }

  _makeWrapper(cls) {
    const d = document.createElement('div');
    d.className = cls;
    return d;
  }

  _makeCanvas(w, h) {
    const c = document.createElement('canvas');
    c.className = 'pr-canvas';
    c.width  = w;
    c.height = h;
    return c;
  }

  // ----------------------------------------------------------------
  // Drawing
  // ----------------------------------------------------------------

  _drawAll() {
    this._drawHeader();
    this._drawGrid();
    this._drawParams();
  }

  _drawHeader() {
    const ctx = this._headerCanvas.getContext('2d');
    ctx.clearRect(0, 0, this._headerCanvas.width, HEADER_H);
    for (let s = 0; s < this._numSteps; s++) {
      const x = s * CELL_W;
      ctx.fillStyle = s === this._currentStep ? '#ff5252' : '#313148';
      ctx.fillRect(x + 1, 1, CELL_W - 2, HEADER_H - 2);
      ctx.fillStyle = s === this._currentStep ? '#fff' : '#8888aa';
      ctx.font = '10px Segoe UI, Arial';
      ctx.textAlign = 'center';
      ctx.fillText(s + 1, x + CELL_W / 2, HEADER_H - 5);
    }
  }

  _drawGrid() {
    const ctx = this._gridCanvas.getContext('2d');
    ctx.clearRect(0, 0, this._gridCanvas.width, this._gridCanvas.height);

    for (let r = 0; r < this._numRows; r++) {
      const rowNote = this._scaleNotes[this._numRows - 1 - r]; // top = highest
      const y = r * CELL_H;
      for (let s = 0; s < this._numSteps; s++) {
        const x = s * CELL_W;
        const rowIdx = this._numRows - 1 - r;
        const active = this._activeNotes[s].has(rowIdx);
        ctx.fillStyle = active
          ? (s === this._currentStep ? '#ff5252' : '#7c6af5')
          : (s === this._currentStep ? '#3a3a5a' : '#2e2e46');
        ctx.fillRect(x + 1, y + 1, CELL_W - 2, CELL_H - 2);
      }
      // Row label (left - we draw it on the grid left edge)
      ctx.fillStyle = '#8888aa';
      ctx.font = '9px Segoe UI, Arial';
      ctx.textAlign = 'left';
      ctx.fillText(rowNote.name, 2, y + CELL_H - 4);
    }
    // Grid lines
    ctx.strokeStyle = '#44445a';
    ctx.lineWidth = 0.5;
    for (let s = 0; s <= this._numSteps; s++) {
      ctx.beginPath();
      ctx.moveTo(s * CELL_W, 0);
      ctx.lineTo(s * CELL_W, this._gridCanvas.height);
      ctx.stroke();
    }
    for (let r = 0; r <= this._numRows; r++) {
      ctx.beginPath();
      ctx.moveTo(0, r * CELL_H);
      ctx.lineTo(this._gridCanvas.width, r * CELL_H);
      ctx.stroke();
    }
  }

  _drawParams() {
    const ctx = this._paramCanvas.getContext('2d');
    ctx.clearRect(0, 0, this._paramCanvas.width, this._paramCanvas.height);

    PARAM_LABELS.forEach((label, rowIdx) => {
      const y = rowIdx * PARAM_ROW_H;
      // Row label
      ctx.fillStyle = '#8888aa';
      ctx.font = '10px Segoe UI, Arial';
      ctx.textAlign = 'left';
      ctx.fillText(label, 2, y + PARAM_ROW_H - 8);

      for (let s = 0; s < this._numSteps; s++) {
        const x = s * CELL_W;
        let val, maxVal, text;
        if (rowIdx === 0) {
          val = this._stepLengths[s];
          text = LEN_VALUES[val];
          maxVal = LEN_VALUES.length - 1;
        } else if (rowIdx === 1) {
          val = this._stepVelocities[s];
          maxVal = 127;
          text = val;
        } else {
          val = this._stepGates[s];
          maxVal = 127;
          text = val;
        }

        // Bar fill (proportional height)
        const fillH = rowIdx === 0
          ? Math.round(((val + 1) / LEN_VALUES.length) * (PARAM_ROW_H - 4))
          : Math.round((val / maxVal) * (PARAM_ROW_H - 4));
        ctx.fillStyle = '#313148';
        ctx.fillRect(x + 1, y + 1, CELL_W - 2, PARAM_ROW_H - 2);
        ctx.fillStyle = '#5a9cf5';
        ctx.fillRect(x + 1, y + PARAM_ROW_H - 2 - fillH, CELL_W - 2, fillH);

        // Text value
        ctx.fillStyle = '#e0e0f0';
        ctx.font = '9px Segoe UI, Arial';
        ctx.textAlign = 'center';
        ctx.fillText(text, x + CELL_W / 2, y + PARAM_ROW_H - 5);
      }
    });

    // Grid lines
    ctx.strokeStyle = '#44445a';
    ctx.lineWidth = 0.5;
    for (let s = 0; s <= this._numSteps; s++) {
      ctx.beginPath();
      ctx.moveTo(s * CELL_W, 0);
      ctx.lineTo(s * CELL_W, this._paramCanvas.height);
      ctx.stroke();
    }
    for (let r = 0; r <= PARAM_ROWS; r++) {
      ctx.beginPath();
      ctx.moveTo(0, r * PARAM_ROW_H);
      ctx.lineTo(this._paramCanvas.width, r * PARAM_ROW_H);
      ctx.stroke();
    }
  }

  // ----------------------------------------------------------------
  // Events
  // ----------------------------------------------------------------

  _attachEvents() {
    // Grid: toggle cells
    this._gridCanvas.addEventListener('click', (e) => {
      const rect = this._gridCanvas.getBoundingClientRect();
      const x = e.clientX - rect.left + this._gridEl.scrollLeft;
      const y = e.clientY - rect.top;
      const step = Math.floor(x / CELL_W);
      const rowVisual = Math.floor(y / CELL_H);
      const rowIdx = this._numRows - 1 - rowVisual;
      if (step < 0 || step >= this._numSteps || rowIdx < 0 || rowIdx >= this._numRows) return;
      if (this._activeNotes[step].has(rowIdx)) {
        this._activeNotes[step].delete(rowIdx);
      } else {
        this._activeNotes[step].add(rowIdx);
      }
      this._drawGrid();
    });

    // Param canvas: drag VEL / GATE rows, click LEN
    let dragRowIdx = null;
    let dragStep   = null;
    let dragStartY = null;
    let dragStartVal = null;

    this._paramCanvas.addEventListener('mousedown', (e) => {
      const rect = this._paramCanvas.getBoundingClientRect();
      const x = e.clientX - rect.left + this._paramEl.scrollLeft;
      const y = e.clientY - rect.top;
      const step = Math.floor(x / CELL_W);
      const rowIdx = Math.floor(y / PARAM_ROW_H);
      if (step < 0 || step >= this._numSteps) return;

      if (rowIdx === 0) {
        // LEN: click cycles through values
        this._stepLengths[step] = (this._stepLengths[step] + 1) % LEN_VALUES.length;
        this._drawParams();
      } else {
        dragRowIdx = rowIdx;
        dragStep   = step;
        dragStartY = e.clientY;
        dragStartVal = rowIdx === 1 ? this._stepVelocities[step] : this._stepGates[step];
      }
    });

    window.addEventListener('mousemove', (e) => {
      if (dragRowIdx === null) return;
      const dy = dragStartY - e.clientY; // drag up = increase
      const newVal = Math.max(0, Math.min(127, dragStartVal + dy));
      if (dragRowIdx === 1) this._stepVelocities[dragStep] = newVal;
      else                  this._stepGates[dragStep]      = newVal;
      this._drawParams();
    });

    window.addEventListener('mouseup', () => {
      dragRowIdx = null;
    });

    // Double-click param to type value
    this._paramCanvas.addEventListener('dblclick', (e) => {
      const rect = this._paramCanvas.getBoundingClientRect();
      const x = e.clientX - rect.left + this._paramEl.scrollLeft;
      const y = e.clientY - rect.top;
      const step = Math.floor(x / CELL_W);
      const rowIdx = Math.floor(y / PARAM_ROW_H);
      if (step < 0 || step >= this._numSteps || rowIdx < 1) return;
      const cur = rowIdx === 1 ? this._stepVelocities[step] : this._stepGates[step];
      const input = window.prompt(`Enter value (0-127) for ${PARAM_LABELS[rowIdx]} step ${step + 1}:`, cur);
      if (input === null) return;
      const v = Math.max(0, Math.min(127, parseInt(input, 10)));
      if (!isNaN(v)) {
        if (rowIdx === 1) this._stepVelocities[step] = v;
        else              this._stepGates[step]      = v;
        this._drawParams();
      }
    });
  }

  // ----------------------------------------------------------------
  // Playback integration
  // ----------------------------------------------------------------

  /** Called by sequencer engine on each step tick. Returns notes to play. */
  advanceStep(step) {
    this._currentStep = step % this._numSteps;
    this._drawHeader();
    this._drawGrid();

    const notes = [];
    this._activeNotes[this._currentStep].forEach((rowIdx) => {
      const note = this._scaleNotes[rowIdx];
      notes.push({
        midi:     note.midi,
        velocity: this._stepVelocities[this._currentStep],
        lenIdx:   this._stepLengths[this._currentStep],
        gate:     this._stepGates[this._currentStep],
      });
    });
    return notes;
  }

  resetPlayhead() {
    this._currentStep = -1;
    this._drawHeader();
    this._drawGrid();
  }

  /** Apply a utility preset pattern. */
  applyUtility(preset, velocity = 100) {
    this._activeNotes = Array.from({ length: this._numSteps }, () => new Set());
    this._stepVelocities = Array(this._numSteps).fill(velocity);
    this._stepLengths    = Array(this._numSteps).fill(0);
    this._stepGates      = Array(this._numSteps).fill(64);

    if (preset === 'up_arp') {
      for (let s = 0; s < this._numSteps; s++) {
        this._activeNotes[s].add(s % this._numRows);
      }
    } else if (preset === 'down_arp') {
      for (let s = 0; s < this._numSteps; s++) {
        this._activeNotes[s].add((this._numRows - 1) - (s % this._numRows));
      }
    } else if (preset === 'pulse_root') {
      for (let s = 0; s < this._numSteps; s++) {
        this._activeNotes[s].add(0);
      }
    }
    // 'empty': already cleared above
    this._drawAll();
  }

  /** Load a musical preset (fixed note indices into scale). */
  loadMusicalPreset(preset) {
    const numSteps = preset.noteIndices.length;
    this._numSteps = numSteps;
    this._activeNotes    = Array.from({ length: numSteps }, () => new Set());
    this._stepVelocities = preset.velocities.slice();
    this._stepLengths    = preset.lengths.slice();
    this._stepGates      = preset.gates.slice();

    preset.noteIndices.forEach((noteIdx, step) => {
      if (noteIdx < this._numRows) {
        this._activeNotes[step].add(noteIdx);
      }
    });
    this._build(); // rebuild canvases with new step count
  }

  /** Update scale (regenerates active note set keeping nearest MIDI notes). */
  updateScale(newScaleNotes) {
    // Collect currently active midi notes
    const activeMidi = new Set();
    for (let s = 0; s < this._numSteps; s++) {
      this._activeNotes[s].forEach((rowIdx) => {
        if (this._scaleNotes[rowIdx]) activeMidi.add(this._scaleNotes[rowIdx].midi);
      });
    }

    this._scaleNotes = newScaleNotes;
    this._numRows = newScaleNotes.length;

    // Rebuild active notes mapping midi -> new rowIdx
    const midiToRow = new Map(newScaleNotes.map((n, i) => [n.midi, i]));
    this._activeNotes = Array.from({ length: this._numSteps }, () => new Set());

    // For each step, remap midi notes to new rows
    for (let s = 0; s < this._numSteps; s++) {
      activeMidi.forEach((midi) => {
        if (midiToRow.has(midi)) {
          this._activeNotes[s].add(midiToRow.get(midi));
        }
      });
    }

    this._build();
  }

  /** Change number of steps (preserves data up to min(old, new) steps). */
  setNumSteps(n) {
    if (n === this._numSteps) return;
    const oldActive    = this._activeNotes;
    const oldVel       = this._stepVelocities;
    const oldLen       = this._stepLengths;
    const oldGate      = this._stepGates;
    this._numSteps     = n;
    this._activeNotes  = Array.from({ length: n }, (_, i) => i < oldActive.length ? new Set(oldActive[i]) : new Set());
    this._stepVelocities = Array.from({ length: n }, (_, i) => i < oldVel.length  ? oldVel[i]  : 100);
    this._stepLengths    = Array.from({ length: n }, (_, i) => i < oldLen.length  ? oldLen[i]  : 0);
    this._stepGates      = Array.from({ length: n }, (_, i) => i < oldGate.length ? oldGate[i] : 64);
    this._build();
  }

  // ----------------------------------------------------------------
  // Save / Load
  // ----------------------------------------------------------------

  toJSON() {
    return {
      version: 1,
      numSteps: this._numSteps,
      scaleNotes: this._scaleNotes,
      notesByStep: this._activeNotes.map((set) => [...set]),
      stepLengths: this._stepLengths,
      stepVelocities: this._stepVelocities,
      stepGates: this._stepGates,
    };
  }

  fromJSON(data) {
    this._numSteps = data.numSteps;
    this._scaleNotes = data.scaleNotes;
    this._numRows    = data.scaleNotes.length;
    this._activeNotes    = data.notesByStep.map((arr) => new Set(arr));
    this._stepLengths    = data.stepLengths;
    this._stepVelocities = data.stepVelocities;
    this._stepGates      = data.stepGates;
    this._build();
  }
}

// ============================================================
// Sequencer
// ============================================================

class Sequencer {
  constructor(midiDevice, transport) {
    this._device    = midiDevice;
    this._transport = transport;
    this._stepCounter = 0;

    // Current scale state
    this._rootName     = 'C';
    this._scaleName    = 'Major';
    this._scaleFamily  = 'Diatonic';
    this._scaleNotes   = [];
    this._startOctave  = 4;  // scroll-able octave offset

    // Piano roll instances
    this._rollStep    = null;  // step sequencer tab
    this._rollPresets = null;  // musical presets tab

    this._activeTab = 'step';

    this._transport.onTick = (tick) => this._onTick(tick);
    this._transport.onStop = () => this._onStop();

    this._buildScaleTree();
    this._buildControls();
    this._updateScale();
  }

  // ----------------------------------------------------------------
  // Scale tree
  // ----------------------------------------------------------------

  _buildScaleTree() {
    const container = document.getElementById('scale-tree');
    container.innerHTML = '';

    Object.entries(SCALE_FAMILIES).forEach(([family, scales]) => {
      const familyDiv = document.createElement('div');

      const header = document.createElement('div');
      header.className = 'scale-family';
      header.innerHTML = `<span>${family}</span><span>▶</span>`;
      familyDiv.appendChild(header);

      const items = document.createElement('div');
      items.className = 'scale-items';
      familyDiv.appendChild(items);

      header.addEventListener('click', () => {
        items.classList.toggle('open');
        header.querySelector('span:last-child').textContent =
          items.classList.contains('open') ? '▼' : '▶';
      });

      Object.keys(scales).forEach((scaleName) => {
        const item = document.createElement('div');
        item.className = 'scale-item';
        item.textContent = scaleName;
        item.dataset.family = family;
        item.dataset.scale  = scaleName;
        item.addEventListener('click', () => this._selectScale(family, scaleName, item));
        items.appendChild(item);
      });

      container.appendChild(familyDiv);
    });

    // Auto-open Diatonic and select Major
    const firstFamily = container.querySelector('.scale-items');
    if (firstFamily) firstFamily.classList.add('open');
    const firstHeader = container.querySelector('.scale-family span:last-child');
    if (firstHeader) firstHeader.textContent = '▼';
    const majorItem = container.querySelector('[data-scale="Major"]');
    if (majorItem) majorItem.classList.add('selected');
  }

  _selectScale(family, scaleName, el) {
    // Deselect all
    document.querySelectorAll('.scale-item').forEach((i) => i.classList.remove('selected'));
    el.classList.add('selected');
    this._scaleFamily = family;
    this._scaleName   = scaleName;
    this._updateScale();
  }

  _updateScale() {
    const intervals = (SCALE_FAMILIES[this._scaleFamily] || {})[this._scaleName];
    if (!intervals) return;
    this._scaleNotes = generateScaleNotes(this._rootName, intervals, 2, this._startOctave);
    const octDisplay = document.getElementById('oct-display');
    if (octDisplay) octDisplay.textContent = `Oct ${this._startOctave}–${this._startOctave + 1}`;
    if (this._rollStep) this._rollStep.updateScale(this._scaleNotes);
  }

  // ----------------------------------------------------------------
  // UI Controls wiring
  // ----------------------------------------------------------------

  _buildControls() {
    // Root note selector
    document.getElementById('root-note').addEventListener('change', (e) => {
      this._rootName = e.target.value;
      this._updateScale();
    });

    // Octave scroll
    document.getElementById('btn-oct-up').addEventListener('click', () => {
      if (this._startOctave < 8) { this._startOctave++; this._updateScale(); }
    });
    document.getElementById('btn-oct-down').addEventListener('click', () => {
      if (this._startOctave > 0) { this._startOctave--; this._updateScale(); }
    });

    // Tab switching
    document.querySelectorAll('.tab-btn').forEach((btn) => {
      btn.addEventListener('click', () => {
        document.querySelectorAll('.tab-btn').forEach((b) => b.classList.remove('active'));
        document.querySelectorAll('.tab-content').forEach((c) => c.classList.remove('active'));
        btn.classList.add('active');
        this._activeTab = btn.dataset.tab;
        document.getElementById(`tab-${btn.dataset.tab}`).classList.add('active');

        // Lazy-init piano rolls
        if (this._activeTab === 'step' && !this._rollStep) {
          this._initStepRoll();
        }
        if (this._activeTab === 'presets' && !this._rollPresets) {
          this._initPresetsRoll();
        }
      });
    });

    // Step count
    document.getElementById('step-count').addEventListener('change', (e) => {
      if (this._rollStep) this._rollStep.setNumSteps(parseInt(e.target.value, 10));
    });

    // Utility preset apply
    document.getElementById('btn-apply-utility').addEventListener('click', () => {
      const preset  = document.getElementById('utility-preset').value;
      const velocity = parseInt(document.getElementById('seq-velocity').value, 10) || 100;
      if (this._rollStep) this._rollStep.applyUtility(preset, velocity);
    });

    // Musical preset apply
    document.getElementById('btn-apply-musical').addEventListener('click', () => {
      const key    = document.getElementById('musical-preset').value;
      const preset = MUSICAL_PRESETS[key];
      if (!preset) return;
      document.getElementById('preset-steps-label').textContent = `Steps: ${preset.numSteps}`;
      this._rootName    = preset.rootName;
      this._scaleName   = preset.scaleName;
      this._scaleFamily = preset.family;
      const intervals   = SCALE_FAMILIES[preset.family][preset.scaleName];
      this._scaleNotes  = generateScaleNotes(preset.rootName, intervals);
      if (!this._rollPresets) this._initPresetsRoll();
      else this._rollPresets.updateScale(this._scaleNotes);
      this._rollPresets.loadMusicalPreset({ ...preset, noteIndices: preset.noteIndices });
    });

    // Save / Load
    document.getElementById('btn-save-pattern').addEventListener('click', () => this._savePattern());
    document.getElementById('btn-load-pattern').addEventListener('click', () => this._loadPattern());

    // Init step roll immediately (default tab)
    this._initStepRoll();
  }

  _initStepRoll() {
    const container = document.getElementById('piano-roll-step');
    const numSteps  = parseInt(document.getElementById('step-count').value, 10) || 16;
    this._rollStep  = new PianoRoll(container, numSteps, this._scaleNotes);
  }

  _initPresetsRoll() {
    const container    = document.getElementById('piano-roll-presets');
    this._rollPresets  = new PianoRoll(container, 32, this._scaleNotes);
  }

  // ----------------------------------------------------------------
  // Tick callback (called by Transport every 1/16 note)
  // ----------------------------------------------------------------

  _onTick(_tick) {
    const roll = this._activeTab === 'step' ? this._rollStep : this._rollPresets;
    if (!roll) return;

    const swing = parseInt(document.getElementById('seq-swing').value, 10) || 0;
    const globalVelocity = parseInt(document.getElementById('seq-velocity').value, 10) || 100;

    // Swing delay on odd steps
    const applySwing = swing > 0 && (this._stepCounter % 2 === 1);
    const swingDelay = applySwing ? Math.round((swing / 100) * (60000 / this._transport.bpm / 4)) : 0;

    const doStep = () => {
      const notes = roll.advanceStep(this._stepCounter);
      notes.forEach((n) => {
        const vel = Math.max(1, Math.min(127, n.velocity !== undefined ? n.velocity : globalVelocity));
        this._device.noteOn(n.midi, vel);

        // Gate duration: 0-127 maps 5% to 400% of a 1/16 base
        const base16ms = (60000 / this._transport.bpm) / 4;
        const gateFactor = 0.05 + (n.gate / 127) * 3.95;
        // Additional length multiplier from LEN setting
        const lenMult = [1, 2, 4][n.lenIdx] || 1;
        const gateMs  = Math.round(gateFactor * base16ms * lenMult);
        setTimeout(() => this._device.noteOff(n.midi), gateMs);
      });
      this._stepCounter++;
    };

    if (swingDelay > 0) {
      setTimeout(doStep, swingDelay);
    } else {
      doStep();
    }
  }

  _onStop() {
    const rollStep    = this._rollStep;
    const rollPresets = this._rollPresets;
    if (rollStep)    rollStep.resetPlayhead();
    if (rollPresets) rollPresets.resetPlayhead();
    this._stepCounter = 0;
  }

  // ----------------------------------------------------------------
  // Save / Load pattern (JSON file download / upload)
  // ----------------------------------------------------------------

  _savePattern() {
    const roll = this._activeTab === 'step' ? this._rollStep : this._rollPresets;
    if (!roll) return;
    const data = JSON.stringify(roll.toJSON(), null, 2);
    const blob = new Blob([data], { type: 'application/json' });
    const a = document.createElement('a');
    a.href = URL.createObjectURL(blob);
    a.download = 'pattern.json';
    a.click();
    URL.revokeObjectURL(a.href);
  }

  _loadPattern() {
    const roll = this._activeTab === 'step' ? this._rollStep : this._rollPresets;
    if (!roll) return;
    const input = document.createElement('input');
    input.type   = 'file';
    input.accept = '.json';
    input.addEventListener('change', () => {
      const file = input.files[0];
      if (!file) return;
      const reader = new FileReader();
      reader.onload = (e) => {
        try {
          const data = JSON.parse(e.target.result);
          roll.fromJSON(data);
        } catch {
          alert('Failed to load pattern: invalid JSON.');
        }
      };
      reader.readAsText(file);
    });
    input.click();
  }
}
