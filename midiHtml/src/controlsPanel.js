/**
 * controlsPanel.js
 * Renders the category tree + parameter sliders/dropdowns/checkboxes.
 * Mirrors ParameterTreeUi + ParameterControlsUi from the Python project.
 */

class ControlsPanel {
  constructor(midiDevice) {
    this._device = midiDevice;
    this._selectedCategory = null;
    this._currentEngineIndex = 0;

    // Persistent state: "category:paramName" -> value
    this._state = {};

    this._elCategoryList  = document.getElementById('category-list');
    this._elParamWidgets  = document.getElementById('param-widgets');
    this._elCategoryTitle = document.getElementById('param-category-title');
    this._elApplyDefaults = document.getElementById('btn-apply-defaults');

    this._buildCategoryTree();
    this._elApplyDefaults.addEventListener('click', () => this._applyDefaults());
  }

  // ----------------------------------------------------------------
  // Category tree
  // ----------------------------------------------------------------

  _buildCategoryTree() {
    this._elCategoryList.innerHTML = '';
    PARAMETER_CATEGORIES.forEach((cat) => {
      const li = document.createElement('li');
      li.textContent = cat.name;
      li.dataset.category = cat.name;
      li.addEventListener('click', () => this._selectCategory(cat.name));
      this._elCategoryList.appendChild(li);
    });
  }

  _selectCategory(name) {
    // Save current state before switching
    this._saveCategoryState();

    // Update selected highlight
    this._elCategoryList.querySelectorAll('li').forEach((li) => {
      li.classList.toggle('selected', li.dataset.category === name);
    });

    this._selectedCategory = name;
    this._elCategoryTitle.textContent = name;
    this._elApplyDefaults.disabled = false;
    this._renderParams(name);
  }

  // ----------------------------------------------------------------
  // Render parameters for a category
  // ----------------------------------------------------------------

  _renderParams(categoryName) {
    this._elParamWidgets.innerHTML = '';
    const cat = PARAMETER_CATEGORIES.find((c) => c.name === categoryName);
    if (!cat) return;

    if (categoryName === 'Sound Engine') {
      this._renderSoundEngine(cat);
    } else {
      cat.params.forEach((p) => this._renderParam(p, categoryName));
    }
  }

  _renderSoundEngine(cat) {
    // First: engine dropdown
    const engineParam = cat.params[0];
    const stateKey = `Sound Engine:Engine`;
    if (!(stateKey in this._state)) this._state[stateKey] = engineParam.default;
    this._currentEngineIndex = this._state[stateKey];
    this._renderDropdown(engineParam, 'Sound Engine', (val) => {
      this._currentEngineIndex = val;
      this._state[stateKey] = val;
      // Re-render engine-specific params below the dropdown
      const engineParamDiv = document.getElementById('engine-specific-params');
      if (engineParamDiv) engineParamDiv.remove();
      this._renderEngineSpecificParams(SOUND_ENGINE_ENGINES[val]);
    });
    this._renderEngineSpecificParams(SOUND_ENGINE_ENGINES[this._currentEngineIndex]);
  }

  _renderEngineSpecificParams(engineName) {
    const container = document.createElement('div');
    container.id = 'engine-specific-params';
    this._elParamWidgets.appendChild(container);

    const params = getEngineParams(engineName);
    params.forEach((p) => {
      const stateKey = `sound_engine:${engineName}:${p.name}`;
      if (!(stateKey in this._state)) this._state[stateKey] = p.default;
      const row = this._buildParamRow(p, this._state[stateKey], (val) => {
        this._state[stateKey] = val;
        this._sendCC(p.cc, val);
      });
      container.appendChild(row);
    });
  }

  _renderParam(p, categoryName) {
    const stateKey = `${categoryName}:${p.name}`;
    if (!(stateKey in this._state)) this._state[stateKey] = p.default;

    if (p.type === 'slider') {
      const row = this._buildSliderRow(p, this._state[stateKey], (val) => {
        this._state[stateKey] = val;
        this._sendCC(p.cc, val);
      });
      this._elParamWidgets.appendChild(row);
    } else if (p.type === 'boolean') {
      const row = this._buildBoolRow(p, this._state[stateKey], (val) => {
        this._state[stateKey] = val;
        this._sendCC(p.cc, val ? 127 : 0);
      });
      this._elParamWidgets.appendChild(row);
    } else if (p.type === 'dropdown') {
      this._renderDropdown(p, categoryName, (val) => {
        this._state[stateKey] = val;
        this._sendCC(p.cc, val);
      });
    }
  }

  _renderDropdown(p, categoryName, onChange) {
    const stateKey = `${categoryName}:${p.name}`;
    if (!(stateKey in this._state)) this._state[stateKey] = p.default;

    const row = document.createElement('div');
    row.className = 'param-row';

    const nameEl = document.createElement('span');
    nameEl.className = 'param-name';
    nameEl.textContent = `${p.name} (CC ${p.cc})`;

    const sel = document.createElement('select');
    (p.options || []).forEach((opt, i) => {
      const o = document.createElement('option');
      o.value = i;
      o.textContent = opt;
      sel.appendChild(o);
    });
    sel.value = this._state[stateKey];

    const valEl = document.createElement('span');
    valEl.className = 'param-value';
    valEl.textContent = this._state[stateKey];

    sel.addEventListener('change', () => {
      const v = parseInt(sel.value, 10);
      valEl.textContent = v;
      onChange(v);
    });

    row.appendChild(nameEl);
    row.appendChild(sel);
    row.appendChild(valEl);
    this._elParamWidgets.appendChild(row);
  }

  // ----------------------------------------------------------------
  // Widget builders
  // ----------------------------------------------------------------

  _buildParamRow(p, currentVal, onChange) {
    if (p.type === 'boolean') return this._buildBoolRow(p, currentVal, onChange);
    return this._buildSliderRow(p, currentVal, onChange);
  }

  _buildSliderRow(p, currentVal, onChange) {
    const row = document.createElement('div');
    row.className = 'param-row';

    const nameEl = document.createElement('span');
    nameEl.className = 'param-name';
    nameEl.textContent = `${p.name} (CC ${p.cc})`;

    const wrap = document.createElement('div');
    wrap.className = 'slider-wrap';

    const slider = document.createElement('input');
    slider.type  = 'range';
    slider.min   = p.min;
    slider.max   = p.max;
    slider.value = currentVal;
    // Live fill via CSS custom property
    slider.style.setProperty('--val', currentVal);
    slider.style.setProperty('--min', p.min);
    slider.style.setProperty('--max', p.max);

    const valEl = document.createElement('span');
    valEl.className = 'param-value';
    valEl.textContent = currentVal;

    slider.addEventListener('input', () => {
      const v = parseInt(slider.value, 10);
      slider.style.setProperty('--val', v);
      valEl.textContent = v;
      onChange(v);
    });

    wrap.appendChild(slider);
    row.appendChild(nameEl);
    row.appendChild(wrap);
    row.appendChild(valEl);
    return row;
  }

  _buildBoolRow(p, currentVal, onChange) {
    const row = document.createElement('div');
    row.className = 'param-bool-row';

    const nameEl = document.createElement('span');
    nameEl.className = 'param-name';
    nameEl.textContent = `${p.name} (CC ${p.cc})`;

    const cb = document.createElement('input');
    cb.type = 'checkbox';
    cb.checked = Boolean(currentVal);

    const label = document.createElement('label');
    label.textContent = 'On';
    label.prepend(cb);

    const valEl = document.createElement('span');
    valEl.className = 'param-value';
    valEl.textContent = currentVal ? 127 : 0;

    cb.addEventListener('change', () => {
      valEl.textContent = cb.checked ? 127 : 0;
      onChange(cb.checked ? 1 : 0);
    });

    row.appendChild(nameEl);
    row.appendChild(label);
    row.appendChild(valEl);
    return row;
  }

  // ----------------------------------------------------------------
  // Apply Defaults
  // ----------------------------------------------------------------

  _applyDefaults() {
    const cat = PARAMETER_CATEGORIES.find((c) => c.name === this._selectedCategory);
    if (!cat) return;

    const sendWithDelay = (params, index) => {
      if (index >= params.length) return;
      const p = params[index];
      this._sendCC(p.cc, p.default);
      const stateKey = `${this._selectedCategory}:${p.name}`;
      this._state[stateKey] = p.default;
      setTimeout(() => sendWithDelay(params, index + 1), 12);
    };

    if (this._selectedCategory === 'Sound Engine') {
      const engineName = SOUND_ENGINE_ENGINES[this._currentEngineIndex];
      const engineParams = getEngineParams(engineName);
      sendWithDelay([cat.params[0], ...engineParams], 0);
      // Re-render to reflect defaults
      setTimeout(() => this._renderParams(this._selectedCategory), 12 * (engineParams.length + 2));
    } else {
      sendWithDelay(cat.params, 0);
      setTimeout(() => this._renderParams(this._selectedCategory), 12 * (cat.params.length + 2));
    }
  }

  // ----------------------------------------------------------------
  // CC send (12 ms inter-message delay handled by caller for bulk)
  // ----------------------------------------------------------------

  _sendCC(cc, value) {
    if (this._device && this._device.isConnected) {
      this._device.controlChange(cc, value);
    }
  }

  // ----------------------------------------------------------------
  // Save category state (sliders update _state directly via closure)
  // ----------------------------------------------------------------

  _saveCategoryState() {
    // State updates happen inline via event listener closures — nothing extra needed.
  }
}
