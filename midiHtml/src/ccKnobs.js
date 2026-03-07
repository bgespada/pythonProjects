/**
 * ccKnobs.js
 * Two tabs of 64 rotary knobs each:
 *   Tab 0: CC  0–63
 *   Tab 1: CC 64–127
 * Interaction: vertical drag, mouse wheel, or double-click to type.
 */

class CcKnobsPanel {
  constructor(midiDevice) {
    this._device = midiDevice;

    // Two banks: bank 0 = CC 0-63, bank 1 = CC 64-127
    this._banks = [
      { startCC: 0,  gridId: 'cc-knobs-grid-0', values: Array(64).fill(0), canvases: [], valLabels: [] },
      { startCC: 64, gridId: 'cc-knobs-grid-1', values: Array(64).fill(0), canvases: [], valLabels: [] },
    ];
    this._activeBank = 0;

    // Single drag state
    this._dragBank     = null;
    this._dragIndex    = null;
    this._dragStartY   = null;
    this._dragStartVal = null;

    this._buildBanks();
    this._buildTabs();
    this._attachGlobalEvents();
  }

  // ----------------------------------------------------------------
  // Build DOM
  // ----------------------------------------------------------------

  _buildBanks() {
    this._banks.forEach((bank, bankIdx) => {
      const container = document.getElementById(bank.gridId);
      container.innerHTML = '';

      for (let i = 0; i < 64; i++) {
        const cc = bank.startCC + i;

        const wrapper = document.createElement('div');
        wrapper.className = 'knob-wrapper';

        const canvas = document.createElement('canvas');
        canvas.width  = 52;
        canvas.height = 52;
        canvas.className = 'knob-canvas';

        const ccLabel = document.createElement('div');
        ccLabel.className = 'knob-cc-label';
        ccLabel.textContent = `CC${cc}`;

        const valLabel = document.createElement('div');
        valLabel.className = 'knob-val-label';
        valLabel.textContent = '0';

        wrapper.appendChild(canvas);
        wrapper.appendChild(ccLabel);
        wrapper.appendChild(valLabel);
        container.appendChild(wrapper);

        bank.canvases[i]  = canvas;
        bank.valLabels[i] = valLabel;

        this._drawKnob(bankIdx, i, 0);
        this._attachKnobEvents(canvas, bankIdx, i);
      }
    });
  }

  _buildTabs() {
    document.querySelectorAll('.cc-tab-btn').forEach((btn) => {
      btn.addEventListener('click', () => {
        document.querySelectorAll('.cc-tab-btn').forEach((b) => b.classList.remove('active'));
        btn.classList.add('active');
        const idx = parseInt(btn.dataset.ccTab, 10);
        this._activeBank = idx;
        this._banks.forEach((bank, i) => {
          document.getElementById(bank.gridId).style.display = i === idx ? '' : 'none';
        });
      });
    });
  }

  _attachKnobEvents(canvas, bankIdx, i) {
    canvas.addEventListener('mousedown', (e) => {
      this._dragBank     = bankIdx;
      this._dragIndex    = i;
      this._dragStartY   = e.clientY;
      this._dragStartVal = this._banks[bankIdx].values[i];
      e.preventDefault();
    });

    // Mouse wheel — 1 unit per notch, 10 units with Shift
    canvas.addEventListener('wheel', (e) => {
      e.preventDefault();
      const step  = e.shiftKey ? 10 : 1;
      const delta = e.deltaY < 0 ? step : -step;
      this._setValue(bankIdx, i, this._banks[bankIdx].values[i] + delta);
    }, { passive: false });

    // Double-click to type exact value
    canvas.addEventListener('dblclick', () => {
      const cc    = this._banks[bankIdx].startCC + i;
      const input = window.prompt(`CC ${cc} — Enter value (0–127):`, this._banks[bankIdx].values[i]);
      if (input === null) return;
      const v = parseInt(input, 10);
      if (!isNaN(v)) this._setValue(bankIdx, i, v);
    });
  }

  _attachGlobalEvents() {
    window.addEventListener('mousemove', (e) => {
      if (this._dragIndex === null) return;
      const dy = this._dragStartY - e.clientY;
      this._setValue(this._dragBank, this._dragIndex, this._dragStartVal + Math.round(dy));
    });

    window.addEventListener('mouseup', () => {
      this._dragBank  = null;
      this._dragIndex = null;
    });
  }

  // ----------------------------------------------------------------
  // Value management
  // ----------------------------------------------------------------

  _setValue(bankIdx, index, raw) {
    const bank = this._banks[bankIdx];
    const v = Math.max(0, Math.min(127, Math.round(raw)));
    if (v === bank.values[index]) return;
    bank.values[index] = v;
    bank.valLabels[index].textContent = v;
    this._drawKnob(bankIdx, index, v);
    this._sendCC(bank.startCC + index, v);
  }

  /** Reset all knobs on the currently visible tab. */
  resetAll() {
    const bank = this._banks[this._activeBank];
    for (let i = 0; i < 64; i++) {
      bank.values[i] = 0;
      bank.valLabels[i].textContent = '0';
      this._drawKnob(this._activeBank, i, 0);
      this._sendCC(bank.startCC + i, 0);
    }
  }

  // ----------------------------------------------------------------
  // Drawing
  // ----------------------------------------------------------------

  _drawKnob(bankIdx, index, value) {
    const canvas = this._banks[bankIdx].canvases[index];
    const ctx    = canvas.getContext('2d');
    const cx     = canvas.width  / 2;
    const cy     = canvas.height / 2 + 2; // slight downward offset for visual balance
    const r      = 17;

    ctx.clearRect(0, 0, canvas.width, canvas.height);

    // Angles: arc runs from 135° to 405° (270° sweep), clockwise
    const startAngle = 135 * Math.PI / 180;
    const sweepAngle = 270 * Math.PI / 180;
    const endAngle   = startAngle + sweepAngle;
    const valueAngle = startAngle + (value / 127) * sweepAngle;

    // Outer ring background
    ctx.beginPath();
    ctx.arc(cx, cy, r + 4, startAngle, endAngle);
    ctx.strokeStyle = '#1e1e2e';
    ctx.lineWidth   = 2;
    ctx.stroke();

    // Track (background arc)
    ctx.beginPath();
    ctx.arc(cx, cy, r, startAngle, endAngle);
    ctx.strokeStyle = '#3a3a5a';
    ctx.lineWidth   = 5;
    ctx.lineCap     = 'round';
    ctx.stroke();

    // Value arc
    if (value > 0) {
      // Colour shifts from blue-purple (low) to bright accent (high)
      const hue = Math.round(240 + (value / 127) * 60); // 240 (blue) → 300 (magenta)
      ctx.beginPath();
      ctx.arc(cx, cy, r, startAngle, valueAngle);
      ctx.strokeStyle = `hsl(${hue}, 80%, 65%)`;
      ctx.lineWidth   = 5;
      ctx.lineCap     = 'round';
      ctx.stroke();
    }

    // Knob body (filled circle)
    ctx.beginPath();
    ctx.arc(cx, cy, r - 4, 0, Math.PI * 2);
    const grad = ctx.createRadialGradient(cx - 3, cy - 3, 1, cx, cy, r - 4);
    grad.addColorStop(0, '#4a4a6a');
    grad.addColorStop(1, '#252535');
    ctx.fillStyle = grad;
    ctx.fill();

    // Indicator dot on knob rim
    const dotX = cx + (r - 6) * Math.cos(valueAngle);
    const dotY = cy + (r - 6) * Math.sin(valueAngle);
    ctx.beginPath();
    ctx.arc(dotX, dotY, 2.5, 0, Math.PI * 2);
    ctx.fillStyle = '#ffffff';
    ctx.fill();
  }

  // ----------------------------------------------------------------
  // MIDI
  // ----------------------------------------------------------------

  _sendCC(cc, value) {
    if (this._device && this._device.isConnected) {
      this._device.controlChange(cc, value);
    }
  }
}
