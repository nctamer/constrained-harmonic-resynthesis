import resynthesis_utils
import silencing_constraints
import numpy as np
import librosa
import pandas as pd
from time import time as taymit
import soundfile as sf
from tqdm import tqdm


HOP_SIZE = 128
SAMPLING_RATE = 44100
WINDOW_SIZE = 1025  # int(2*(((1024/16000)*SAMPLING_RATE)//2))-1
WINDOW_TYPE = 'blackmanharris'
LOWEST_NOTE_ALLOWED_HZ = 180


def analyze_file(paths, confidence_threshold=0.9, min_voiced_segment_ms=25):
    time_start = taymit()
    filename = ' '.join(paths['original'].split('/')[-3:])[:-4]
    audio = librosa.load(paths['original'], sr=SAMPLING_RATE, mono=True)[0]
    f0s = pd.read_csv(paths['f0'])
    f0s, conf, time = resynthesis_utils.interpolate_f0_to_sr(f0s, audio)
    time_load = taymit()
    print("loading {:s} took {:.3f}".format(filename, time_load - time_start))
    hfreqs, hmags, _ = resynthesis_utils.anal(audio, f0s, n_harmonics=12)
    f0s = f0s[:len(hmags)]
    conf = conf[:len(hmags)]
    time = time[:len(hmags)]
    time_anal = taymit()

    conf_bool = conf > confidence_threshold
    conf_bool_1 = conf < 1.0
    conf_bool = np.logical_and(conf_bool, conf_bool_1)
    valid_f0_bool = f0s > LOWEST_NOTE_ALLOWED_HZ
    # lowest note on violin is G3 = 196 hz, so threshold with sth close to the lowest note
    valid_hmag_bool = (hmags > -100).sum(axis=1) > 3  # at least three harmonics
    valid_bool = np.logical_and(valid_hmag_bool, valid_f0_bool)
    valid_bool = np.logical_and(conf_bool, valid_bool)
    min_voiced_segment_len = int(np.ceil((min_voiced_segment_ms / 1000) / (HOP_SIZE / SAMPLING_RATE)))
    valid_bool = silencing_constraints.silence_segments_one_run(valid_bool, 0, min_voiced_segment_len)  # if keeps high for some duration

    print("anal {:s} took {:.3f}. coverage: {:.3f}".format(filename, time_anal - time_load,
                                                           sum(valid_bool) / len(valid_bool)))
    np.savez_compressed(paths['anal'], f0=f0s[valid_bool], hmag=hmags[valid_bool, :12])
    return


def synth_file(paths, instrument_detector=None, refine_twm=True, pitch_shift=False,
               th_lc=0.2, th_hc=0.7, voiced_th_ms=100, sawtooth_synth=False, create_tfrecords=False):
    time_start = taymit()
    audio = librosa.load(paths['original'], sr=SAMPLING_RATE, mono=True)[0]
    f0s = pd.read_csv(paths['f0'])
    filename = ' '.join(paths['original'].split('/')[-3:])[:-4]
    f0s["confidence"] = f0s["confidence"].fillna(0)
    pre_anal_coverage = f0s['confidence'] > th_lc
    pre_anal_coverage = sum(pre_anal_coverage) / len(pre_anal_coverage)
    f0s = silencing_constraints.silence_unvoiced_segments(f0s, low_confidence_threshold=th_lc,
                                                          high_confidence_threshold=th_hc,
                                                          min_voiced_segment_ms=voiced_th_ms)
    # f0s = apply_pitch_filter(f0s, min_chunk_size=21, median=True, confidence_threshold=th_hc)
    f0s, conf, time = resynthesis_utils.interpolate_f0_to_sr(f0s, audio)
    time_load = taymit()
    print("loading {:s} took {:.3f}".format(filename, time_load - time_start))
    hfreqs, hmags, hphases = resynthesis_utils.anal(audio, f0s, n_harmonics=40)
    f0s = f0s[:len(hmags)]
    conf = conf[:len(hmags)]
    time = time[:len(hmags)]
    time_anal = taymit()
    print("anal {:s} took {:.3f}".format(filename, time_anal - time_load))
    if instrument_detector is not None:
        hfreqs, hmags, hphases, f0 = silencing_constraints.supress_timbre_anomalies(instrument_detector,
                                                                                    hfreqs, hmags, hphases, f0s)
    if refine_twm:
        hfreqs, hmags, hphases, f0s = silencing_constraints.refine_harmonics_twm(hfreqs, hmags, hphases,
                                                                                 f0s, f0et=5.0,
                                                                                 f0_refinement_range_cents=16,
                                                                                 min_voiced_segment_ms=voiced_th_ms)
    time_refine = taymit()
    post_anal_coverage = sum(f0s > 0) / len(f0s)
    coverage = post_anal_coverage / pre_anal_coverage
    print("refining parameters for {:s} took {:.3f}. coverage: {:.3f}".format(filename,
                                                                              time_refine - time_anal,
                                                                              coverage))
    if sawtooth_synth:
        hmags[f0s > 0] = -30 - 20 * np.log10(np.arange(1, 41))
        hfreqs[f0s > 0] = np.dot(hfreqs[f0s > 0][:, 0][:, np.newaxis], np.arange(1, 41)[np.newaxis, :])
        hphases = np.array([])
    harmonic_audio = resynthesis_utils.synth(hfreqs, hmags, hphases, N=512, H=HOP_SIZE, fs=SAMPLING_RATE)
    sf.write(paths['synth'], harmonic_audio, 44100, 'PCM_24')
    df = pd.DataFrame([time, f0s]).T
    df.to_csv(paths['synth_f0'], header=False, index=False,
              float_format='%.6f')
    if create_tfrecords:
        tfrecord_file(paths, 'synth')
    if pitch_shift:
        sign = random.choice([-1, 1])
        val = random.choice(range(5, 50))
        pitch_shift_cents = sign * val

        alt_f0s = f0s * pow(2, (pitch_shift_cents / 1200))
        # Synthesize audio with the shifted harmonic content
        alt_hfreqs = hfreqs * pow(2, (pitch_shift_cents / 1200))
        alt_harmonic_audio = resynthesis_utils.synth(alt_hfreqs, hmags, np.array([]), N=512,
                                                     H=HOP_SIZE, fs=SAMPLING_RATE)
        sf.write(paths['shifted'], alt_harmonic_audio,
                 44100, 'PCM_24')
        df = pd.DataFrame([time, alt_f0s]).T
        df.to_csv(paths['shifted_f0'], header=False, index=False, float_format='%.6f')
        if create_tfrecords:
            tfrecord_file(paths, 'shifted')

    time_synth = taymit()
    print("synthesizing {:s} took {:.3f}. Total resynthesis took {:.3f}".format(filename, time_synth - time_refine,
                                                                                time_synth - time_load))
    return


def tfrecord_file(paths, name='synth'):
    import tensorflow.compat.v1 as tf
    options = tf.python_io.TFRecordOptions(tf.python_io.TFRecordCompressionType.GZIP)
    labels = np.loadtxt(paths[name+'_f0'], delimiter=',')

    nonzero = labels[:, 1] > 0
    absdiff = np.abs(np.diff(np.concatenate(([False], nonzero, [False]))))
    ranges = np.where(absdiff == 1)[0].reshape(-1, 2)
    nonzero[ranges[:, 0]] = False
    nonzero[ranges[:, 1]-1] = False
    # get rid of the boundary points since it may contain some artifacts
    # since the hop size in synthesis is 2.9 ms it roughly corresponds to 512/16000
    labels = labels[nonzero, :]

    if len(labels):
        sr = 16000
        audio = librosa.load(paths[name], sr=sr)[0]

        output_path = paths[name+'_tfrecord']
        writer = tf.python_io.TFRecordWriter(output_path, options=options)

        for row in tqdm(labels):
            pitch = row[1]
            center = int(row[0] * sr)
            segment = audio[center - 512:center + 512]
            if len(segment) == 1024:
                example = tf.train.Example(features=tf.train.Features(feature={
                    "audio": tf.train.Feature(float_list=tf.train.FloatList(value=segment)),
                    "pitch": tf.train.Feature(float_list=tf.train.FloatList(value=[pitch]))
                }))
                writer.write(example.SerializeToString())
        writer.close()
    return

