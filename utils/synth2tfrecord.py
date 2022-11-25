import os
import tensorflow.compat.v1 as tf
import librosa
from tqdm import tqdm
from joblib import Parallel, delayed
import numpy as np



options = tf.python_io.TFRecordOptions(tf.python_io.TFRecordCompressionType.GZIP)

def process_file(stem, path_folder_synth, path_folder_tfrecord):
    labels = np.loadtxt(os.path.join(path_folder_synth, stem + '.csv'), delimiter=',')

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
        audio = librosa.load(os.path.join(path_folder_synth, stem + '.wav'), sr=sr)[0]

        output_path = os.path.join(path_folder_tfrecord, stem + '.tfrecord')
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


def process_folder(path_folder_synth, path_folder_tfrecord, n_jobs=4):
    if not os.path.exists(path_folder_tfrecord):
        # Create a new directory because it does not exist
        os.makedirs(path_folder_tfrecord)
    stems = [_[:-4] for _ in sorted(os.listdir(path_folder_synth)) if _.endswith('.wav')]
    stem_check = [_[:-4] for _ in sorted(os.listdir(path_folder_synth)) if _.endswith('.csv')]
    try:
        assert stems == stem_check
    except AssertionError:
        print('File name mismatch in folder', path_folder_synth)
        pass

    Parallel(n_jobs=n_jobs)(delayed(process_file)(
        stem, path_folder_synth, path_folder_tfrecord) for stem in stems)


if __name__ == '__main__':
    names = ["L1", "L2", "L3", "L4", "L5", "L6"]
    dataset_folder = os.path.join(os.path.expanduser("~"), "violindataset", "graded_repertoire")

    # '/home/nazif/violindataset/graded_repertoire/tfrecord/L1/S1_BochanKang_015_15. Minuet 3.RESYN.tfrecord'
    # PROBLEM!!

    for name in names:
        print("started processing the folder ", name)

        process_folder(path_folder_synth=os.path.join(dataset_folder, "synthesized", name),
                       path_folder_tfrecord=os.path.join(dataset_folder, "tfrecord", name),
                       n_jobs=16)
