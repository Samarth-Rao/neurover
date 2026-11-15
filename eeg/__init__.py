"""Everything EEG that both apps share (the rover in main.py, neurover scope in scope.py):

    devices.py      live BrainBits (or a mock), each on its own clock, with raw + filtered buffers
    brainbit.py     the BrainBit SDK thread behind each device
    mock.py         synthetic EEG (one virtual head; each device adds its own reference noise)
    filters.py      streaming band-pass + notch, spectra
    clean.py        noise handling: judging electrodes, down-weighting spikes
    ssvep.py        SSVEP decoding + calibration
    recording.py    the .npz recording format
"""
