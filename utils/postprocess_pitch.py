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

def silence_segments_one_run(confidences, confidence_threshold, segment_len_th):
    conf_bool = np.array(confidences > confidence_threshold).reshape(-1)
    absdiff = np.abs(np.diff(np.concatenate(([False], conf_bool, [False]))))
    ranges = np.where(absdiff == 1)[0].reshape(-1, 2)
    segment_durs = np.diff(ranges, axis=1)
    valid_segments = ranges[np.repeat(segment_durs > segment_len_th, repeats=2, axis=1)].reshape(-1, 2)
    voiced = np.zeros(len(confidences), dtype=bool)
    for segment in valid_segments:
        voiced[segment[0]:segment[1]] = True
    return voiced


def silence_unvoiced_segments(pitch_track_csv, low_confidence_threshold=0.2,
                              high_confidence_threshold=0.7, min_voiced_segment_ms=12):
    """
    Accepts crepe (or any other pitch estimator) output in the csv format and removes unvoiced segments based on two confidence thresholds.
    Removes pitch estimates with confidences below the low threshold. Accepts estimates with confidences above the high threshold.
    If the confidence is in between high and low values, then apply a median filtering with the voiced_segment_ms parameter. 
    The same segment duration is also used to silence short voiced segments, which sound like arbitrary noises.
    :param pitch_track_csv: csv with [ºtimeº, ºfrequencyº, ºconfidenceº] fields
    :param low_confidence_threshold: confidence threshold in range (0,1). 
    :param high_confidence_threshold: confidence threshold in range (0,1)/
    :param min_voiced_segment_ms: voiced segments shorter than the specified lenght are discarded.
    :return: input csv file with the silenced segments
    """
    annotation_interval_ms = 1000 * pitch_track_csv.loc[:1, "time"].diff()[1]
    voiced_th = int(np.ceil(min_voiced_segment_ms / annotation_interval_ms))

    # we do not accept the segment if a close neighbors do not have a confidence > 0.7
    smoothened_confidences = medfilt(pitch_track_csv["confidence"], kernel_size=2 * (voiced_th // 2) + 1)
    smooth_voiced = silence_segments_one_run(smoothened_confidences,
                                             confidence_threshold=high_confidence_threshold, segment_len_th=voiced_th)

    # we also do not accept the pitch values if the individual confidences are really low
    hard_voiced = silence_segments_one_run(pitch_track_csv["confidence"],
                                           confidence_threshold=low_confidence_threshold, segment_len_th=voiced_th)

    # we accept the intersection of these two zones
    voiced = np.logical_and(smooth_voiced, hard_voiced)

    smoothened_pitch = copy.deepcopy(pitch_track_csv["frequency"])
    smoothened_pitch[~voiced] = np.nan
    smoothened_pitch.fillna(smoothened_pitch.rolling(window=15, min_periods=8).median(), inplace=True)

    # medfilt(pitch_track_csv["frequency"], kernel_size=21)
    absdiff = np.abs(np.diff(np.concatenate(([False], voiced, [False]))))
    ranges = np.where(absdiff == 1)[0].reshape(-1, 2)
    unvoiced_ranges = np.vstack([ranges[:-1, 1], ranges[1:, 0]]).T
    for unvoiced_boundary in unvoiced_ranges:
        # we don't want small unvoiced zones. Check if they are acceptable with a more favorable mean thresholding
        len_unvoiced = np.diff(unvoiced_boundary)[0]
        if len_unvoiced < voiced_th:
            avg_confidence = pitch_track_csv.loc[unvoiced_boundary[0]:unvoiced_boundary[1], "confidence"].mean()
            if avg_confidence > low_confidence_threshold:
                voiced[unvoiced_boundary[0]:unvoiced_boundary[1]] = True
            elif len_unvoiced < 8:
                # and (unvoiced_boundary[0] > 3) and (unvoiced_boundary[-1] < len(pitch_track_csv)-3):
                pitch_track_csv.loc[unvoiced_boundary[0]:unvoiced_boundary[1], "frequency"] = \
                    smoothened_pitch[unvoiced_boundary[0]:unvoiced_boundary[1]]
                voiced[unvoiced_boundary[0]:unvoiced_boundary[1]] = True


    pitch_track_csv.loc[~voiced, "frequency"] = 0
    pitch_track_csv["frequency"] = pitch_track_csv["frequency"].fillna(0)
    return pitch_track_csv

