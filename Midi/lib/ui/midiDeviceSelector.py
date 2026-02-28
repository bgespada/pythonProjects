import tkinter as tk
from tkinter import ttk, messagebox
from typing import Optional, Callable
from pathlib import Path
import sys

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from soundCard.midiDiscoverer import MidiDiscoverer


class MidiDeviceSelector:
    """
    A tkinter-based GUI class for selecting MIDI input and/or output devices.
    Uses MidiDiscoverer to discover available devices.
    """
    
    def __init__(self, 
                 parent=None,
                 device_type: str = 'both',
                 callback: Optional[Callable[[str], None]] = None,
                 title: str = "MIDI Device Selector"):
        """
        Initialize MidiDeviceSelector.
        
        Args:
            parent: Parent tkinter widget (None creates root window)
            device_type (str): 'input', 'output', or 'both'
            callback (Optional[Callable]): Function to call when device is selected
            title (str): Window title
        """
        self.device_type = device_type
        self.callback = callback
        self.selected_device: Optional[str] = None
        self.discoverer = MidiDiscoverer()
        
        # Create window based on parent
        if parent is None:
            self.root = tk.Tk()
            self.is_root_owner = True
            self.is_toplevel = False
        else:
            self.root = tk.Toplevel(parent)
            self.is_root_owner = False
            self.is_toplevel = True
            # Make window modal
            self.root.transient(parent)
            self.root.grab_set()
        
        self.root.title(title)
        self.root.geometry("500x400")
        self.root.resizable(True, True)
        
        self._create_widgets()
        self._load_devices()
    
    def _create_widgets(self) -> None:
        """Create the GUI widgets."""
        # Main frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Configure grid weights for responsiveness
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(0, weight=1)
        main_frame.rowconfigure(2, weight=1)
        
        # Title label
        title_label = ttk.Label(
            main_frame,
            text=f"Select MIDI Device ({self.device_type.upper()})",
            font=("Arial", 12, "bold")
        )
        title_label.grid(row=0, column=0, columnspan=2, sticky=tk.W, pady=(0, 10))
        
        # Device type selection (if 'both' is selected)
        if self.device_type == 'both':
            filter_frame = ttk.LabelFrame(main_frame, text="Device Type", padding="5")
            filter_frame.grid(row=1, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 10))
            filter_frame.columnconfigure(0, weight=1)
            
            self.device_filter = tk.StringVar(value='output')
            
            ttk.Radiobutton(
                filter_frame,
                text="Input Devices",
                variable=self.device_filter,
                value='input',
                command=self._load_devices
            ).pack(side=tk.LEFT, padx=5)
            
            ttk.Radiobutton(
                filter_frame,
                text="Output Devices",
                variable=self.device_filter,
                value='output',
                command=self._load_devices
            ).pack(side=tk.LEFT, padx=5)
            
            row_offset = 2
        else:
            self.device_filter = tk.StringVar(value=self.device_type)
            row_offset = 1
        
        # Listbox with scrollbar
        list_frame = ttk.LabelFrame(main_frame, text="Available Devices", padding="5")
        list_frame.grid(row=row_offset, column=0, columnspan=2, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(0, 10))
        list_frame.columnconfigure(0, weight=1)
        list_frame.rowconfigure(0, weight=1)
        
        # Scrollbar
        scrollbar = ttk.Scrollbar(list_frame)
        scrollbar.grid(row=0, column=1, sticky=(tk.N, tk.S))
        
        # Listbox
        self.device_listbox = tk.Listbox(
            list_frame,
            yscrollcommand=scrollbar.set,
            height=10,
            font=("Arial", 10)
        )
        self.device_listbox.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        scrollbar.config(command=self.device_listbox.yview)
        
        # Bind double-click event
        self.device_listbox.bind('<Double-Button-1>', self._on_device_double_click)
        
        # Button frame
        button_frame = ttk.Frame(main_frame)
        button_frame.grid(row=row_offset + 1, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 10))
        button_frame.columnconfigure(0, weight=1)
        button_frame.columnconfigure(1, weight=1)
        button_frame.columnconfigure(2, weight=1)
        
        # Refresh button
        refresh_btn = ttk.Button(button_frame, text="Refresh", command=self._load_devices)
        refresh_btn.grid(row=0, column=0, sticky=(tk.W, tk.E), padx=(0, 5))
        
        # Select button
        select_btn = ttk.Button(button_frame, text="Select", command=self._on_select)
        select_btn.grid(row=0, column=1, sticky=(tk.W, tk.E), padx=(0, 5))
        
        # Cancel button
        cancel_btn = ttk.Button(button_frame, text="Cancel", command=self._on_cancel)
        cancel_btn.grid(row=0, column=2, sticky=(tk.W, tk.E))
        
        # Status label
        self.status_label = ttk.Label(main_frame, text="", foreground="blue")
        self.status_label.grid(row=row_offset + 2, column=0, columnspan=2, sticky=tk.W)
    
    def _load_devices(self) -> None:
        """Load and display MIDI devices based on selected type."""
        self.device_listbox.delete(0, tk.END)
        
        device_type = self.device_filter.value if hasattr(self.device_filter, 'value') else self.device_filter.get()
        
        if device_type == 'input' or self.device_type == 'input':
            devices = self.discoverer.get_input_devices()
            device_label = "Input"
        else:
            devices = self.discoverer.get_output_devices()
            device_label = "Output"
        
        if devices:
            for device in devices:
                self.device_listbox.insert(tk.END, device)
            self.status_label.config(
                text=f"Found {len(devices)} {device_label.lower()} device(s)",
                foreground="green"
            )
        else:
            self.status_label.config(
                text=f"No {device_label.lower()} devices found",
                foreground="red"
            )
    
    def _on_device_double_click(self, event) -> None:
        """Handle double-click on device in listbox."""
        self._on_select()
    
    def _on_select(self) -> None:
        """Handle select button click."""
        selection = self.device_listbox.curselection()
        
        if not selection:
            messagebox.showwarning("No Selection", "Please select a device from the list!")
            return
        
        self.selected_device = self.device_listbox.get(selection[0])
        
        if self.callback:
            self.callback(self.selected_device)
        
        self.status_label.config(
            text=f"Selected: {self.selected_device}",
            foreground="green"
        )
        
        if self.is_root_owner:
            self.root.after(500, self.root.quit)
        else:
            # For Toplevel windows, just close the window
            self.root.destroy()
    
    def _on_cancel(self) -> None:
        """Handle cancel button click."""
        self.selected_device = None
        if self.is_root_owner:
            self.root.quit()
        else:
            # For Toplevel windows, just close the window
            self.root.destroy()
    
    def get_selected_device(self) -> Optional[str]:
        """
        Get the selected device.
        
        Returns:
            Optional[str]: Selected device name or None
        """
        return self.selected_device
    
    def show(self) -> Optional[str]:
        """
        Display the window and return selected device (blocking).
        Only works when this class owns the root window.
        
        Returns:
            Optional[str]: Selected device name or None
        """
        if self.is_root_owner:
            self.root.mainloop()
            return self.selected_device
        else:
            # For Toplevel windows, wait for it to close
            self.root.wait_window()
            return self.selected_device
    
    def close(self) -> None:
        """Close the window."""
        if self.is_root_owner:
            self.root.quit()
        else:
            self.root.destroy()


def open_midi_device_selector(device_type: str = 'both') -> Optional[str]:
    """
    Convenience function to open MIDI device selector dialog.
    
    Args:
        device_type (str): 'input', 'output', or 'both'
    
    Returns:
        Optional[str]: Selected device name or None
    """
    selector = MidiDeviceSelector(device_type=device_type)
    return selector.show()


# if __name__ == "__main__":
#     # Test the selector
#     selected = open_midi_device_selector(device_type='output')
    
#     if selected:
#         print(f"Selected device: {selected}")
#     else:
#         print("No device selected")
