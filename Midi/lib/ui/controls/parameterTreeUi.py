import tkinter as tk
from tkinter import ttk
from typing import Callable, Optional
from midi.ccParameters import PARAMETER_CATEGORIES


class ParameterTreeUi:
    """
    Left-side tree widget that lists parameter categories and their parameters.
    Calls on_select(category_name) when the user selects a category.
    """

    def __init__(self, parent: tk.Widget, on_select: Callable[[str], None]):
        """
        Args:
            parent: Parent tkinter widget.
            on_select: Callback called with the selected category name.
        """
        self.on_select = on_select

        # Container frame
        self.frame = ttk.Frame(parent)

        # Scrollbar + Treeview
        scrollbar = ttk.Scrollbar(self.frame, orient=tk.VERTICAL)
        self.tree = ttk.Treeview(
            self.frame,
            yscrollcommand=scrollbar.set,
            selectmode="browse",
            show="tree",          # hide header column
            height=10,
        )
        scrollbar.config(command=self.tree.yview)

        self.tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        self._populate()
        self.tree.bind("<<TreeviewSelect>>", self._on_tree_select)

    def _populate(self) -> None:
        """Populate the tree with categories from PARAMETER_CATEGORIES."""
        for category in PARAMETER_CATEGORIES:
            self.tree.insert("", tk.END, iid=category, text=category, open=False)

    def _on_tree_select(self, event) -> None:
        """Handle treeview selection and call on_select callback."""
        selected = self.tree.focus()
        if selected and selected in PARAMETER_CATEGORIES:
            self.on_select(selected)

    def grid(self, **kwargs) -> None:
        self.frame.grid(**kwargs)

    def pack(self, **kwargs) -> None:
        self.frame.pack(**kwargs)
