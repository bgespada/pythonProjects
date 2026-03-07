/**
 * midiDevice.js
 * Manages Web MIDI API access, device enumeration, connection, and low-level send.
 */

class MidiDevice {
  constructor() {
    this._access = null;
    this._output = null;
    this._deviceName = null;
    this.channel = 0;           // 0-based (0-15)
    this.defaultVelocity = 64;
    this.defaultOctave = 5;
    this.noteOffMode = 'note_off'; // 'note_off' | 'zero_velocity'

    // Callbacks
    this.onConnect = null;
    this.onDisconnect = null;
    this.onStateChange = null;
  }

  /** Request MIDI access. Returns true on success. */
  async requestAccess() {
    if (!navigator.requestMIDIAccess) {
      throw new Error('Web MIDI API is not supported in this browser. Use Chrome or Edge.');
    }
    this._access = await navigator.requestMIDIAccess({ sysex: false });
    this._access.onstatechange = (e) => this._handleStateChange(e);
    return true;
  }

  /** List all available MIDI output devices. Returns array of {id, name}. */
  listOutputs() {
    if (!this._access) return [];
    const result = [];
    for (const [id, port] of this._access.outputs) {
      result.push({ id, name: port.name });
    }
    return result;
  }

  /** Connect to a MIDI output by id. */
  connect(deviceId, { channel = 0, velocity = 64, octave = 5, noteOffMode = 'note_off' } = {}) {
    if (!this._access) throw new Error('MIDI access not initialised.');
    const port = this._access.outputs.get(deviceId);
    if (!port) throw new Error(`Device not found: ${deviceId}`);
    this._output = port;
    this._deviceName = port.name;
    this.channel = channel;
    this.defaultVelocity = velocity;
    this.defaultOctave = octave;
    this.noteOffMode = noteOffMode;
    if (this.onConnect) this.onConnect(port.name);
  }

  /** Disconnect from the current output device. */
  disconnect() {
    this._output = null;
    this._deviceName = null;
    if (this.onDisconnect) this.onDisconnect();
  }

  get isConnected() { return this._output !== null; }
  get deviceName()  { return this._deviceName; }

  // ------------------------------------------------------------------
  // Low-level send helpers
  // ------------------------------------------------------------------

  _send(data) {
    if (this._output) this._output.send(data);
  }

  /** Send Note On. Channel is 0-based. */
  noteOn(note, velocity = null, ch = null) {
    const c = (ch !== null ? ch : this.channel) & 0x0F;
    const v = Math.max(0, Math.min(127, velocity !== null ? velocity : this.defaultVelocity));
    this._send([0x90 | c, note & 0x7F, v]);
  }

  /** Send Note Off (or note_on zero velocity depending on mode). */
  noteOff(note, velocity = 0, ch = null) {
    const c = (ch !== null ? ch : this.channel) & 0x0F;
    if (this.noteOffMode === 'zero_velocity') {
      this._send([0x90 | c, note & 0x7F, 0]);
    } else {
      this._send([0x80 | c, note & 0x7F, Math.max(0, Math.min(127, velocity))]);
    }
  }

  /** Send Control Change. */
  controlChange(control, value, ch = null) {
    const c = (ch !== null ? ch : this.channel) & 0x0F;
    this._send([0xB0 | c, control & 0x7F, Math.max(0, Math.min(127, value))]);
  }

  /** Send Program Change. */
  programChange(program, ch = null) {
    const c = (ch !== null ? ch : this.channel) & 0x0F;
    this._send([0xC0 | c, program & 0x7F]);
  }

  /** Send Pitch Bend. pitch is 0-16383 (8192 = centre). */
  pitchBend(pitch = 8192, ch = null) {
    const c = (ch !== null ? ch : this.channel) & 0x0F;
    const p = Math.max(0, Math.min(16383, pitch));
    this._send([0xE0 | c, p & 0x7F, (p >> 7) & 0x7F]);
  }

  /** Send All Notes Off (CC 123, value 0) — useful on stop. */
  allNotesOff(ch = null) {
    const c = (ch !== null ? ch : this.channel) & 0x0F;
    this._send([0xB0 | c, 123, 0]);
  }

  // MIDI clock / transport (real-time messages, no channel byte)
  sendStart()    { this._send([0xFA]); }
  sendStop()     { this._send([0xFC]); }
  sendContinue() { this._send([0xFB]); }
  sendClock()    { this._send([0xF8]); }

  // ------------------------------------------------------------------
  // Internal
  // ------------------------------------------------------------------

  _handleStateChange(e) {
    if (e.port.type !== 'output') return;
    if (this._output && e.port.id === this._output.id) {
      if (e.port.state === 'disconnected') {
        this.disconnect();
      }
    }
    if (this.onStateChange) this.onStateChange(e);
  }
}
