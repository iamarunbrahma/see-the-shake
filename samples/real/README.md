# Real test clips

Real machines filmed by other people, under open licenses, for testing until Arun's own Redmi clips arrive.

## aeg-washer-120fps

`aeg-washer-1400rpm-120fps.webm`: an AEG front-loading washing machine during a 1400 rpm spin, filmed on an Android phone in 120 fps slow motion.

- Source: "Washing machine 120fps" by Golha, Wikimedia Commons,
  https://commons.wikimedia.org/wiki/File:Washing_machine_120fps.webm
- License: CC BY-SA 4.0 (https://creativecommons.org/licenses/by-sa/4.0/). Anything we make from it, such as magnified
  versions, must keep the credit and be shared under CC BY-SA 4.0.
- Change: trimmed to 8 to 18.5 s of the original 26.9 s with a stream copy, without re-encoding. The file keeps the original's
  `com.android.capture.fps=120` tag.
- Notes:
  - Video: 1920x1080 at 120 fps with sound.
  - Handheld: the background shakes at 6 to 12 Hz, typical hand tremor.
  - A quick check of the control panel against the background shows a steady peak at 24.0 Hz (about 1440 rpm) through
    the whole original, with a 2x peak at 48 Hz.
  - The machine is a standard 600 mm wide front-loader.

## xjtu-rotor-500fps

A single-disk, dual-span rotor test rig at Xi'an Jiaotong University, filmed with a Phantom high-speed camera.
Four eddy-current displacement probes recorded the shaft at the same time.

- Source: Kang, Zheng, Liu, Pei, "Vibration Displacement Dataset of Rotors under Various Operational Conditions:
  High-speed Camera Video Data and Eddy current displacement sensor data", Mendeley Data, V1, 2024,
  doi:10.17632/47y7rwc2yg.1
- License: CC BY 4.0 (https://creativecommons.org/licenses/by/4.0/).
- Change: none. This folder holds the 9 steady-state runs only (1000, 2000 and 3000 rpm, each with unbalance masses
  Unb1 to Unb3). The 4 s and 6 s run-ups (about 70 MB each) are left out.
- Files:
  - `<rpm>-<Unb>.avi`: 500 frames at 1280x800. The file says 100 fps, but the camera captured them at 500 fps, so each
    clip covers 1.0 s of real time.
  - `<rpm>-<Unb>.csv`: four columns, one per probe position, sampled at 1000 per second. The probe sensitivity is
    10 V/mm. The CSV length doesn't match the video, so the two aren't time-aligned; compare the spectra instead.
  - `README.md`: the dataset authors' own notes. At 2000 rpm, probes 2 and 3 went out of range; don't use them.
  - `Frame-TestRig.jpg`, `Rotor-TestRig.jpg`, `Rotor-TestRig.STEP`: a camera frame, a photo and the 3D model
    (for real dimensions, such as the disc diameter, to set the scale).
