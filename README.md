# wire-maze-biosonar
Designing Wire Mazes for Replicating Natural Echoes to Study Bat Biosonar Function

Chunlin Jia, Gengkai Hu, Khoo Boo Cheong, Rolf Müller*

The code takes frequency-domain scattering data produced by the Multiple Scattering Model (MSM), reconstructs time-domain echoes, converts them into spectrograms, and trains a CNN to classify five types of scatterer spatial distributions.

Repository structure:

nlfm_pulse.py — NLFM pulse generation (see Supporting Information S2)

generate_spectrograms.py — echo reconstruction from MSM data and STFT spectrogram computation

prepare_dataset.py — Jet colormap encoding and train/val/test splitting

train_cnn.py — CNN model definition, training, and evaluation

How to use:

The workflow has three steps. First, generate spectrograms from MSM scattering data (one file per distribution type):
python generate_spectrograms.py 

Then prepare the dataset (applies Jet colormap and splits into train/val/test):
python prepare_dataset.py

Finally, train and evaluate the CNN:
python train_cnn.py

Signal parameters:

The default NLFM pulse parameters match those in the paper (Section 2.3.3): center frequency 70 kHz, bandwidth 20 kHz, pulse duration 2 ms, and sweep-shaping parameter n = 20. These can be modified in generate_spectrograms.py.

Requirements:

numpy, scipy, tensorflow (>= 2.10), scikit-learn, matplotlib, hdf5storage
pip install numpy scipy tensorflow scikit-learn matplotlib hdf5storage
