import numpy as np

def ApplyDistortion(waveform, distortion_type="clipping", amount=0.8):
    """
    Apply distortion to an existing waveform.

    Parameters:
        waveform (np.ndarray): The original waveform (1D array).
        distortion_type (str): Type of distortion ("clipping", "wavefolding", "nonlinear").
        amount (float): Amount of distortion (0.0 to 1.0).

    Returns:
        np.ndarray: Distorted waveform of the same size.
    """
    if distortion_type == "clipping":
        # Apply hard clipping
        return np.clip(waveform, -amount, amount)
    elif distortion_type == "wavefolding":
        # Apply wavefolding distortion
        folded = (np.abs(waveform) % (2 * amount)) - amount
        return folded
    elif distortion_type == "nonlinear":
        # Apply nonlinear distortion (e.g., tanh)
        return np.tanh(waveform * amount * 5)
    else:
        return waveform
        # raise ValueError(f"Unknown distortion type: {distortion_type}")

