import copy
from tqdm import tqdm
import pandas as pd
from scipy import interpolate
from scipy.signal import get_window, medfilt
from utils.pitchfilter import PitchFilter
import sys
import pickle
import random
from pathlib import Path
import glob
import os
import tensorflow.compat.v1 as tf
import librosa
import numpy as np

def refine_harmonics_twm(hfreq, hmag, hphases, f0, f0et=5.0, f0_refinement_range_cents=10, min_voiced_segment_ms=100):
    """
    Refine the f0 estimate with the help of two-way mismatch algorithm and change the harmonic components
    to the exact multiples of the refined f0 estimate
    :param hfreq: analyzed harmonic frequencies
    :param hmag: analyzed magnitudes
    :param f0: f0 in Hz before TWM
    :param f0et: error threshold for the TWM
    :param f0_refinement_range_cents: the range to be explored in TWM
    :return: new synthesis parameters
    """
    for frame, f0_frame in enumerate(f0):
        if f0_frame > 0:  # for the valid frequencies
            pfreq = hfreq[frame]
            pmag = hmag[frame]
            f0_twm, f0err_twm = refinef0Twm(pfreq, pmag, f0_frame, refinement_range_cents=f0_refinement_range_cents)
            if f0err_twm < f0et:
                hfreq[frame] = f0_twm * np.round(pfreq / f0_twm)
                f0[frame] = f0_twm
            else:
                f0[frame] = 0
                hfreq[frame] = 0
                hmag[frame] = -100

    min_voiced_segment_len = int(np.ceil((min_voiced_segment_ms / 1000) / (HOP_SIZE / SAMPLING_RATE)))
    voiced = silence_segments_one_run(f0, 0, min_voiced_segment_len)
    f0[~voiced] = 0
    hfreq[~voiced] = 0
    hmag[~voiced] = -100
    hphases[~voiced] = 0
    return hfreq, hmag, hphases, f0


def supress_timbre_anomalies(instrument_detector, hfreq, hmag, hphase, f0):
    hmag_ptr = np.copy(hmag)[:, 1:12]
    hmag_ptr[np.isnan(hmag_ptr)] = -100
    hvalid = np.logical_and(hmag_ptr[:, 0] > -100, f0 > 0)
    hmag_ptr[hvalid] = hmag_ptr[hvalid] - hmag[hvalid, 0][:, np.newaxis]
    filter_centroids = np.array(list(instrument_detector.keys()))
    f0_classes = np.abs(f0[:,np.newaxis] - filter_centroids[np.newaxis,:]).argmin(axis=1)
    relevant_classes = np.unique(f0_classes[hvalid])
    voiced = hvalid
    for pitch_class in relevant_classes:
        relevant_timbre_model = instrument_detector[filter_centroids[pitch_class]]
        relevant_section = np.logical_and(f0_classes==pitch_class, hvalid)
        voiced_relevant = relevant_timbre_model.predict(hmag_ptr[relevant_section])
        voiced[relevant_section] = voiced_relevant > 0

    f0[~voiced] = 0
    hfreq[~voiced] = 0
    hmag[~voiced] = -100
    hphase[~voiced] = 0
    return hfreq, hmag, hphase, f0
