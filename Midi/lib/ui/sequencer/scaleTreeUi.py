import tkinter as tk
from tkinter import ttk
from typing import Callable
from midi.scales import SCALE_FAMILIES, ROOT_NOTES, generate_notes, note_name


class ScaleTreeUi:
    """
    Left-side panel with:
      - Root note selector (combobox)
      - Treeview of scale families and their individual scales

    Calls on_scale_select(midi_notes, note_names) when a scale is selected
    or the root note changes.
    """

    ROOTS = list(ROOT_NOTES.keys())

    def __init__(self, parent: tk.Widget, on_scale_select: Callable[[list[int], list[str]], None]):
        """
        Args:
            parent: Parent tkinter widget.
            on_scale_select: Callback receiving (midi_notes, note_names) lists.
        """
        self.on_scale_select = on_scale_select
        self._selected_family: str | None = None
        self._selected_scale: str | None = None
        self._iid_to_scale: dict[str, tuple[str, str]] = {}

        self.frame = ttk.Frame(parent)

        # Root note selector
        root_frame = ttk.Frame(self.frame)
        root_frame.pack(fill=tk.X, padx=4, pady=(4, 4))
        ttk.Label(root_frame, text="Root:", font=("Arial", 9)).pack(side=tk.LEFT)
        self._root_var = tk.StringVar(value="C")
        root_cb = ttk.Combobox(
            root_frame,
            textvariable=self._root_var,
            values=self.ROOTS,
            width=4,
            state="readonly",
        )
        root_cb.pack(side=tk.LEFT, padx=(4, 0))
        root_cb.bind("<<ComboboxSelected>>", self._on_root_change)

        # Treeview + scrollbar
        tree_frame = ttk.Frame(self.frame)
        tree_frame.pack(fill=tk.BOTH, expand=True, padx=4, pady=(0, 4))

        scrollbar = ttk.Scrollbar(tree_frame, orient=tk.VERTICAL)
        self.tree = ttk.Treeview(
            tree_frame,
            yscrollcommand=scrollbar.set,
            selectmode="browse",
            show="tree",
            height=10,
        )
        scrollbar.config(command=self.tree.yview)
        self.tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        self._populate()
        self.tree.bind("<<TreeviewSelect>>", self._on_tree_select)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def grid(self, **kwargs) -> None:
        self.frame.grid(**kwargs)

    def pack(self, **kwargs) -> None:
        self.frame.pack(**kwargs)

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------

    def _populate(self) -> None:
        for fam_idx, (family, scales) in enumerate(SCALE_FAMILIES.items()):
            fam_iid = f"fam_{fam_idx}"
            self.tree.insert("", tk.END, iid=fam_iid, text=family, open=True)
            for scale_idx, scale_name in enumerate(scales):
                scale_iid = f"sca_{fam_idx}_{scale_idx}"
                self._iid_to_scale[scale_iid] = (family, scale_name)
                self.tree.insert(fam_iid, tk.END, iid=scale_iid, text=scale_name)

    def _on_tree_select(self, _event=None) -> None:
        selected = self.tree.focus()
        if selected in self._iid_to_scale:
            self._selected_family, self._selected_scale = self._iid_to_scale[selected]
            self._emit()

    def _on_root_change(self, _event=None) -> None:
        if self._selected_family and self._selected_scale:
            self._emit()

    def _emit(self) -> None:
        if not (self._selected_family and self._selected_scale):
            return
        root_midi = ROOT_NOTES.get(self._root_var.get(), 48)
        intervals = SCALE_FAMILIES[self._selected_family][self._selected_scale]
        notes = generate_notes(root_midi, intervals, octaves=2)
        names = [note_name(n) for n in notes]
        self.on_scale_select(notes, names)
