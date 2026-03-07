/**
 * transport.js
 * MIDI transport controls: Start / Pause / Stop + MIDI clock ticker.
 * Also exposes global tempo (BPM) used by the sequencer engine.
 */

class Transport {
  constructor(midiDevice) {
    this._device = midiDevice;
    this._bpm = 120;
    this._clockInterval = null;   // setInterval handle for MIDI clock (24 ppqn)
    this._running = false;
    this.onTick = null;           // called every 1/16th note by sequencer engine
    this._tickCount = 0;          // counts MIDI clocks; 6 clocks = 1 sixteenth
  }

  get bpm() { return this._bpm; }
  set bpm(value) {
    this._bpm = Math.max(30, Math.min(300, value));
    if (this._running) {
      this._stopClock();
      this._startClock();
    }
  }

  get isRunning() { return this._running; }

  start() {
    if (!this._device.isConnected) return;
    this._device.sendStart();
    this._tickCount = 0;
    this._running = true;
    this._startClock();
  }

  pause() {
    if (!this._device.isConnected) return;
    this._device.sendContinue();
    // Keep clock running on pause (continue mode)
  }

  stop() {
    if (!this._device.isConnected) return;
    this._device.sendStop();
    this._running = false;
    this._stopClock();
    this._device.allNotesOff();
    if (this.onStop) this.onStop();
  }

  _startClock() {
    // MIDI clock = 24 pulses per quarter note
    const intervalMs = (60000 / this._bpm) / 24;
    this._clockInterval = setInterval(() => {
      this._device.sendClock();
      this._tickCount++;
      // 6 MIDI clocks = 1 sixteenth note
      if (this._tickCount % 6 === 0 && this.onTick) {
        this.onTick(this._tickCount);
      }
    }, intervalMs);
  }

  _stopClock() {
    if (this._clockInterval !== null) {
      clearInterval(this._clockInterval);
      this._clockInterval = null;
    }
  }
}
