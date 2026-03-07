/**
 * main.js — Entry point for the MIDI HTML application.
 * Wires together: MidiDevice, Transport, ControlsPanel, Sequencer, and UI.
 */

// All classes loaded via <script> tags in index.html

// ============================================================
// State
// ============================================================

const midi      = new MidiDevice();
let transport   = null;
let controls    = null;
let sequencer   = null;

// ============================================================
// Elements
// ============================================================

const elDeviceName    = document.getElementById('device-name');
const elBtnSelect     = document.getElementById('btn-select-device');
const elBtnDisconnect = document.getElementById('btn-disconnect');
const elBtnStart      = document.getElementById('btn-start');
const elBtnPause      = document.getElementById('btn-pause');
const elBtnStop       = document.getElementById('btn-stop');
const elTempoInput    = document.getElementById('tempo-input');
const elStatusInfo    = document.getElementById('status-info');
const elStatusChannel = document.getElementById('status-channel');

// Modal elements
const elOverlay        = document.getElementById('modal-overlay');
const elDeviceList     = document.getElementById('device-list');
const elBtnRefresh     = document.getElementById('btn-modal-refresh');
const elBtnModalSelect = document.getElementById('btn-modal-select');
const elBtnModalCancel = document.getElementById('btn-modal-cancel');
const elModalStatus    = document.getElementById('modal-status');
const elCfgChannel     = document.getElementById('cfg-channel');
const elCfgVelocity    = document.getElementById('cfg-velocity');
const elCfgVelVal      = document.getElementById('cfg-vel-val');
const elCfgOctave      = document.getElementById('cfg-octave');
const elCfgOctVal      = document.getElementById('cfg-oct-val');
const elCfgNoteOff     = document.getElementById('cfg-noteoff-mode');

// ============================================================
// Status helpers
// ============================================================

function setStatus(msg, type = '') {
  elStatusInfo.textContent = msg;
  elStatusInfo.className   = 'status-info' + (type ? ` ${type}` : '');
}

function setChannelStatus(ch) {
  elStatusChannel.textContent = ch >= 0
    ? `MIDI Channel: ${ch + 1}`
    : 'MIDI Channel: All';
}

// ============================================================
// Connection management
// ============================================================

function onConnected(deviceName) {
  elDeviceName.textContent = deviceName;
  elDeviceName.className   = 'device-name connected';
  elBtnDisconnect.disabled = false;
  elBtnStart.disabled      = false;
  elBtnPause.disabled      = false;
  elBtnStop.disabled       = false;
  setStatus(`Connected to: ${deviceName}`, 'ok');
  setChannelStatus(midi.channel);
}

function onDisconnected() {
  elDeviceName.textContent = 'None';
  elDeviceName.className   = 'device-name disconnected';
  elBtnDisconnect.disabled = true;
  elBtnStart.disabled      = true;
  elBtnPause.disabled      = true;
  elBtnStop.disabled       = true;
  elStatusChannel.textContent = '';
  setStatus('Disconnected');
  if (transport && transport.isRunning) transport.stop();
}

midi.onConnect    = onConnected;
midi.onDisconnect = onDisconnected;

// ============================================================
// Modal: device selector
// ============================================================

function populateDeviceList() {
  elDeviceList.innerHTML = '';
  elBtnModalSelect.disabled = true;
  elModalStatus.textContent = '';

  const outputs = midi.listOutputs();
  if (outputs.length === 0) {
    const opt = document.createElement('option');
    opt.textContent = '(no MIDI output devices found)';
    opt.disabled = true;
    elDeviceList.appendChild(opt);
    elModalStatus.textContent = 'No MIDI output devices found. Check connections.';
    return;
  }

  outputs.forEach(({ id, name }) => {
    const opt = document.createElement('option');
    opt.value       = id;
    opt.textContent = name;
    elDeviceList.appendChild(opt);
  });
}

elCfgVelocity.addEventListener('input', () => {
  elCfgVelVal.textContent = elCfgVelocity.value;
});
elCfgOctave.addEventListener('input', () => {
  elCfgOctVal.textContent = elCfgOctave.value;
});
elDeviceList.addEventListener('change', () => {
  elBtnModalSelect.disabled = elDeviceList.value === '';
});

elBtnSelect.addEventListener('click', async () => {
  // Request MIDI access if not yet done
  if (!midi._access) {
    try {
      await midi.requestAccess();
    } catch (err) {
      setStatus(`MIDI Error: ${err.message}`, 'error');
      return;
    }
  }
  populateDeviceList();
  elOverlay.classList.remove('hidden');
});

elBtnRefresh.addEventListener('click', populateDeviceList);

elBtnModalSelect.addEventListener('click', () => {
  const id = elDeviceList.value;
  if (!id) return;
  try {
    midi.connect(id, {
      channel:     parseInt(elCfgChannel.value, 10),
      velocity:    parseInt(elCfgVelocity.value, 10),
      octave:      parseInt(elCfgOctave.value, 10),
      noteOffMode: elCfgNoteOff.value,
    });
    elOverlay.classList.add('hidden');
  } catch (err) {
    elModalStatus.textContent = `Error: ${err.message}`;
  }
});

elBtnModalCancel.addEventListener('click', () => {
  elOverlay.classList.add('hidden');
});

// Close modal on overlay click
elOverlay.addEventListener('click', (e) => {
  if (e.target === elOverlay) elOverlay.classList.add('hidden');
});

elBtnDisconnect.addEventListener('click', () => {
  if (transport && transport.isRunning) transport.stop();
  midi.disconnect();
});

// ============================================================
// Transport
// ============================================================

transport = new Transport(midi);

elTempoInput.addEventListener('change', () => {
  transport.bpm = parseInt(elTempoInput.value, 10) || 120;
});
elTempoInput.addEventListener('input', () => {
  transport.bpm = parseInt(elTempoInput.value, 10) || 120;
});

elBtnStart.addEventListener('click', () => transport.start());
elBtnPause.addEventListener('click', () => transport.pause());
elBtnStop.addEventListener('click',  () => transport.stop());

// ============================================================
// Controls panel
// ============================================================

controls = new ControlsPanel(midi);

// ============================================================
// Sequencer
// ============================================================

sequencer = new Sequencer(midi, transport);

// ============================================================
// CC Knobs
// ============================================================

const ccKnobs = new CcKnobsPanel(midi);
document.getElementById('btn-cc-reset-all').addEventListener('click', () => ccKnobs.resetAll());

// ============================================================
// Initial status
// ============================================================

setStatus('Not connected — click "Select" to choose a MIDI device');
