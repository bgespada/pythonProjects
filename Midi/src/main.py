"""
Main entry point for the MIDI application.
"""
from pathlib import Path
import sys

# Add parent directory to path so we can import lib
sys.path.insert(0, str(Path(__file__).parent.parent))

from lib.ui.ui import MidiUI


def main():
    """Start the MIDI application."""
    app = MidiUI()
    app.run()


if __name__ == "__main__":
    main()
