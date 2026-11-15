# attic

Code that is not part of the app right now. Nothing here is imported, run or tested (pytest only
looks in `tests/`). It is kept so it can be brought back. The imports still point at the old layout
(`sources/`, `pipeline/`, `sinks/`), and the whole old app is in git history at commit `c48e1ef`.

* `myo/`: driving with the Myo armband (EMG). The Bluegiga dongle and Bluetooth workers, the mock arm,
  the windowing + KNN gesture classifier and the training panel (T, hold 0-4, Enter, Ctrl+S).
* `alpha/`: alpha mode (eyes closed = forward). Occipital 8-12 Hz power over 4-30 Hz, with a
  30 s eyes-open / eyes-closed calibration.
* `tests/`: their tests and the fake pyomyo.

To bring one back, port it onto the current layout:
* `rover/source.py`: Source, Prediction (label, confidence, timestamp, scores)
* `eeg/devices.py`: live EEG
* `rover/controller.py`
