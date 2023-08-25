"""
1. Synthesize audio data
2. Feature extraction for audio data.
"""
import os
import time
import argparse
import re
import multiprocessing
import logging
import random
import numpy as np
import wandb
import boto3
import soundfile as sf
import sounddevice as sd
import librosa
from nnsp_pack import tfrecord_converter_esc
from nnsp_pack.feature_module import FeatureClass, display_stft_all
from nnsp_pack import add_noise
from nnsp_pack import boto3_op
from nnsp_pack.esc_download import esc_download
from nnsp_pack.basic_dsp import dc_remove
import matplotlib.pyplot as plt

DEBUG = False
UPLOAD_TFRECORD_S3 = False
DOWLOAD_DATA = False
REVERB = True
FRAMES_TARGET_DELAY = 20
TARGETS=[
    "unknown",
    "dog_bark",
    "gun_shot",
    "siren_car_horn",
    "baby_crying"]

if UPLOAD_TFRECORD_S3:
    print('uploading tfrecords to s3 will slow down the process')
S3_BUCKET = "ambiqai-speech-commands-dataset"
S3_PREFIX = "tfrecords"
if DEBUG:
    SNR_DBS = [ 6]
else:
    SNR_DBS = [3, 6, 9, 12, 15, 30]

NTYPES = [
    'ESC-50-MASTER',
    'wham_noise',
    'FSD50K',
    'musan',
]
params_audio = {
    'win_size'      : 240,
    'hop'           : 80,
    'len_fft'       : 256,
    'sample_rate'   : 8000,
    'nfilters_mel'  : 22 }

def download_data():
    """
    download data
    """
    audio_lists = [
        'data/test_files_se.csv',
        'data/train_files_se.csv',
        'data/noise_list.csv']
    s3 = boto3.client('s3')
    boto3_op.s3_download(S3_BUCKET, audio_lists)
    return s3

class FeatMultiProcsClass(multiprocessing.Process):
    """
    FeatMultiProcsClass use multiprocesses
    to run several processes of feature extraction in parallel
    """
    def __init__(self, id_process,
                 name, src_list, train_set, success_dict,
                 params_audio_def,
                 num_procs = 8,
                 reverb_lst=None,
                 reverb_prob=0,
                 wavs_sp=[],
                 wavs_noise=[]
                 ):

        multiprocessing.Process.__init__(self)
        self.success_dict = success_dict
        self.id_process         = id_process
        self.name               = name
        self.src_list           = src_list
        self.params_audio_def   = params_audio_def
        self.num_procs          = num_procs
        self.reverb_lst         = reverb_lst
        self.reverb_prob        = reverb_prob
        self.feat_inst      = FeatureClass(
                                win_size        = params_audio_def['win_size'],
                                hop             = params_audio_def['hop'],
                                len_fft         = params_audio_def['len_fft'],
                                sample_rate     = params_audio_def['sample_rate'],
                                nfilters_mel    = params_audio_def['nfilters_mel'])

        self.train_set      = train_set
        self.wavs_sp        = wavs_sp
        self.wavs_noise     = wavs_noise
        self.names=[]
        if DEBUG:
            self.cnt = 0

    def run(self):
        #      threadLock.acquire()
        print("Running " + self.name)
        random.shuffle(self.src_list)
        self.convert_tfrecord(
                    self.src_list,
                    self.id_process)

    def audio_set_len(self, audio, length):
        """_summary_

        Args:
            audio (_type_): _description_
            length (_type_): _description_

        Returns:
            _type_: _description_
        """
        if len(audio) < length:
            tmp = np.zeros(length)
            s = np.random.randint(length - len(audio))
            tmp[s:s+len(audio)] = audio
            audio = tmp

        elif len(audio) > length:
            s = np.random.randint(len(audio) - length)
            audio = audio[s:s+length]
        return audio

    def audio_rep_len(self, audio, length):
        """_summary_

        Args:
            audio (_type_): _description_
            length (_type_): _description_

        Returns:
            _type_: _description_
        """
        rep = length // len(audio) + 1
        audio = np.tile(audio, rep)
        audio = audio[:length]
        return audio

    def audio_load(
            self,
            wavname,
            target_sr = 8000):
        """_summary_

        Args:
            wavname (_type_): _description_
            sampling_rate (_type_): _description_

        Returns:
            _type_: _description_
        """
        audio, sample_rate = sf.read(wavname)
        if audio.ndim > 1:
            audio=audio[:,0]
        if sample_rate > target_sr:
            audio = librosa.resample(
                    audio,
                    orig_sr=sample_rate,
                    target_sr=target_sr)
        return audio

    def convert_tfrecord(
            self,
            fnames,
            id_process):
        """
        convert np array to tfrecord
        """
        GAINS=np.arange(0, 1.0, 0.05) + 0.01
        for r in range(2):
            if r == 0:
                reverbing = False
            else:
                reverbing = True
            for i_gain in range(len(GAINS)-1):
                random.shuffle(fnames)
                random.shuffle(self.wavs_sp)
                random.shuffle(self.wavs_noise)
                reverbing=True
                for i in range(len(fnames) >> 1):
                    if self.num_procs-1 == self.id_process:
                        print(f"\rProcessing wav {i}/{len(fnames) >> 1}", end="")
                    success = 1
                    stimes = []
                    etimes = []
                    targets = []
                    speech = np.empty(0)
                    len_sp_last = 0
                    pattern = r'(\.wav$|\.flac$)'

                    start_filler = 0
                    for k in range(2):
                        fname = fnames[2*i+k]
                        wavpath, stime, etime, target_name = fname.strip().split(',')
                        stime = int(stime)
                        etime = int(etime)
                        target = TARGETS.index(target_name)
                        if k == 0:
                            tfrecord = re.sub(
                                pattern,
                                '.tfrecord',
                                re.sub(r'wavs', S3_PREFIX, wavpath))
                        try:
                            audio, sample_rate = sf.read(wavpath)
                        except :# pylint: disable=bare-except
                            success = 0
                            print(f"Reading the {wavpath} fails ")
                        else:
                            audio = audio[stime:etime]

                            if audio.ndim > 1:
                                audio=audio[:,0]

                            if len(audio) > 6 * sample_rate:
                                audio = audio[:6 * sample_rate]

                            if sample_rate > self.feat_inst.sample_rate:
                                audio = librosa.resample(
                                        audio,
                                        orig_sr=sample_rate,
                                        target_sr=self.feat_inst.sample_rate)
                            # decorate speech
                            amp_sig = np.random.uniform(0.5, 2)
                            audio = audio / (np.abs(audio).max() + 10**-5)
                            speech_raw = audio * amp_sig

                            stime = np.random.randint(
                                int(self.feat_inst.sample_rate * 1.0),
                                self.feat_inst.sample_rate * 3)
                            if k == 0:
                                pp = np.random.uniform(0, 1)
                                if pp < 0.25:
                                    stime = 1
                            zeros_s = np.zeros(stime)

                            size_zeros = np.random.randint(
                                int(self.feat_inst.sample_rate * 1.0),
                                self.feat_inst.sample_rate * 3)

                            zeros_e = np.zeros(size_zeros)
                            etime = len(speech_raw) + stime
                            speech0 = np.concatenate((zeros_s, speech_raw, zeros_e))

                            # read filler noise
                            noise = self.audio_load(
                                self.wavs_noise[(2*i+k) % len(self.wavs_noise)],
                                self.feat_inst.sample_rate)

                            noise = self.audio_rep_len(
                                        noise,
                                        self.feat_inst.sample_rate * 12)
                            amp_sig = np.random.uniform(0.1, 0.95)
                            filler = noise * amp_sig
                            if np.random.uniform(0, 1) < 0.1:
                                amp_n = np.random.uniform(0.0, 0.95)
                                filler = np.random.randn(len(filler)) * amp_n * 0.01
                            noise_filler = self.audio_set_len(filler, len(speech0))
                            random.shuffle(SNR_DBS)
                            speech0, _ = add_noise.add_noise(
                                        speech0,
                                        noise_filler,
                                        SNR_DBS[0],
                                        stime=0, etime=len(speech0),
                                        return_all=True,
                                        snr_dB_improved = None,
                                        rir=None,
                                        min_amp=0.5,
                                        max_amp=0.95)

                            stime += FRAMES_TARGET_DELAY * self.feat_inst.hop
                            etime += FRAMES_TARGET_DELAY * self.feat_inst.hop # delay 50 frames to label quiet
                            stimes += [stime + len_sp_last]
                            etimes += [etime + len_sp_last]

                            if np.random.uniform(0, 1) < 0.5:
                                # # read filler speech
                                speech_filler = self.audio_load(
                                    self.wavs_sp[(2*i+k) % len(self.wavs_sp)],
                                    self.feat_inst.sample_rate)
                                random.shuffle(SNR_DBS)
                                speech_filler = self.audio_set_len(speech_filler, len(speech0))
                                speech0, _ = add_noise.add_noise(
                                        speech0,
                                        speech_filler,
                                        SNR_DBS[0],
                                        stime=0, etime=len(speech0),
                                        return_all=True,
                                        snr_dB_improved = None,
                                        rir=None,
                                        min_amp=0.5,
                                        max_amp=0.95)

                            if np.random.uniform(0,1) < 0.25:
                                amp_n = np.random.uniform(0.0, 0.95)
                                speech0 = np.random.randn(len(speech0)) * amp_n * 0.01 # silence
                                target = 0
                            targets += [target]
                            speech = np.concatenate((speech, speech0))

                            len_sp_last += len(speech)

                    stimes  = np.array(stimes)
                    etimes  = np.array(etimes)

                    targets = np.array(targets).astype(np.int32)
                    start_frames    = (stimes / self.params_audio_def['hop']) + 1 # target level frame
                    start_frames    = start_frames.astype(np.int32)
                    end_frames      = (etimes / self.params_audio_def['hop']) + 1 # target level frame
                    end_frames      = end_frames.astype(np.int32)
                    # add noise to sig
                    rir = None

                    if self.reverb_lst:
                        idx = np.random.randint(0, len(self.reverb_lst))

                        rir, sample_rate_rir = sf.read(self.reverb_lst[idx])

                        if rir.ndim > 1:
                            rir = rir[:,0]
                        if sample_rate_rir > self.feat_inst.sample_rate:
                            rir = librosa.resample(
                                    rir,
                                    orig_sr=sample_rate_rir,
                                    target_sr=self.feat_inst.sample_rate)
                        rir = rir[:np.minimum(3000, rir.size)]
                    # add reverb
                    # speech = dc_remove(speech)
                    audio_sn = speech
                    if reverbing:
                        audio_sn = np.convolve(audio_sn, rir, 'same')

                    amp_sig = np.random.uniform(GAINS[i_gain], GAINS[i_gain+1])
                    audio_sn = audio_sn / (np.abs(audio_sn).max() + 10**-5) * amp_sig

                    # feature extraction of sig
                    spec_sn, _, feat_sn, pspec_sn = self.feat_inst.block_proc(audio_sn)

                    if DEBUG:
                        if reverbing:
                            print('has reverb')
                        
                        sd.play(
                            audio_sn,
                            self.feat_inst.sample_rate)
                        print(fnames[2*i])
                        print(fnames[2*i + 1])
                        print(start_frames)
                        flabel = np.zeros(spec_sn.shape[0])
                        for start_frame, end_frame, target in zip(start_frames, end_frames, targets):
                            flabel[start_frame: end_frame] = target

                        display_stft_all(
                            audio_sn, spec_sn.T, feat_sn.T,
                            audio_sn, spec_sn.T, feat_sn.T,
                            self.feat_inst.sample_rate,
                            start_frames, end_frames, targets)
                        print(f"{TARGETS[targets[0]]}, {TARGETS[targets[1]]}")
                        
                        os.makedirs('test_wavs', exist_ok=True)
                        sf.write(f'test_wavs/speech_{self.cnt}.wav',
                                audio_sn,
                                self.feat_inst.sample_rate)

                        sf.write(f'test_wavs/speech_{self.cnt}_ref.wav',
                                speech,
                                self.feat_inst.sample_rate)

                        self.cnt = self.cnt + 1

                    if success:
                        if reverbing:
                            tfrecord = re.sub(  r'\.tfrecord$',
                                                f'_gain{int(i_gain)}_reverb.tfrecord',
                                                tfrecord)
                        else:
                            tfrecord = re.sub(  r'\.tfrecord$',
                                                f'_gain{int(i_gain)}.tfrecord',
                                                tfrecord)
                        os.makedirs(os.path.dirname(tfrecord), exist_ok=True)
                        try:
                            timesteps, _  = feat_sn.shape
                            width_targets = end_frames - start_frames + 1
                            if not DEBUG:
                                tfrecord_converter_esc.make_tfrecord( # pylint: disable=too-many-function-args
                                                    tfrecord,
                                                    feat_sn,
                                                    targets,
                                                    timesteps,
                                                    start_frames,
                                                    width_targets)

                        except: # pylint: disable=bare-except
                            print(f"Thread-{id_process}: {i}, processing {tfrecord} failed")
                        else:
                            self.success_dict[self.id_process] += [tfrecord]
                            # since tfrecord file starts: data/tfrecords/speakers/...
                            # strip the leading "data/" when uploading
                            if UPLOAD_TFRECORD_S3:
                                s3.upload_file(tfrecord, S3_BUCKET, tfrecord)
                            else:
                                pass
def parse_wavs(files_sp, start=1):
    wavs_sp = {"train": [], "test": []}
    for file_sp in files_sp:
        for i, set0 in enumerate(["train", "test"]):
            with open(file_sp[i], 'r') as file:
                lines = file.readlines()
                for line in lines[start:]:
                    fname = line.strip().split(',')[0]
                    wavs_sp[set0] += [fname]

    return wavs_sp

def main(args):
    """
    main function to generate all training and testing data
    """
    download = args.download
    datasize_noise = args.datasize_noise
    reverb_prob = args.reverb_prob
    if download:
        esc_download()

    if DOWLOAD_DATA:
        s3 = download_data()
    if args.wandb_track:
        run = wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            job_type="data-update")
        wandb.config.update(args)
    folders_sp =[["data/train_speech.csv", "data/test_speech.csv"]]
    wavs_sp =parse_wavs(folders_sp)

    folders_noise =[
        ["data/train_noise_fsd50k.csv", "data/test_noise_fsd50k.csv"],
        ["data/train_noise_musan.csv", "data/test_noise_musan.csv"],
        ["data/train_noise_wham.csv", "data/test_noise_wham.csv"]]
    wavs_noise =parse_wavs(folders_noise, start=0)


    sets_categories = ['train', 'test']

    if REVERB:
        tmp = add_noise.get_noise_files_new("rirs_noises/RIRS_NOISES/simulated_rirs")
        random.shuffle(tmp)
        start = int(len(tmp) / 5)
        lst_reverb = {}
        lst_reverb = {'train': tmp[start:],
                      'test': tmp[:start]}
    else:
        lst_reverb = None
    target_files = { 'train': args.train_dataset_path,
                     'test' : args.test_dataset_path}
    tot_success_dict = {'train': [], 'test': []}
    for train_set in sets_categories:

        with open(target_files[train_set], 'r') as file: # pylint: disable=unspecified-encoding
            filepaths = file.readlines()
            random.shuffle(filepaths)

        blk_size = int(np.floor(len(filepaths) / args.num_procs))
        sub_src = []
        for i in range(args.num_procs):
            idx0 = i * blk_size
            if i == args.num_procs - 1:
                sub_src += [filepaths[idx0:]]
            else:
                sub_src += [filepaths[idx0:blk_size+idx0]]

        manager = multiprocessing.Manager()
        success_dict = manager.dict({i: [] for i in range(args.num_procs)})
        print(f'{train_set} set running:')
        processes = [
            FeatMultiProcsClass(
                    i, f"Thread-{i}",
                    sub_src[i],
                    train_set,
                    success_dict,
                    params_audio_def = params_audio,
                    num_procs = args.num_procs,
                    reverb_lst = lst_reverb[train_set],
                    reverb_prob = reverb_prob,
                    wavs_sp = wavs_sp[train_set],
                    wavs_noise = wavs_noise[train_set])
                        for i in range(args.num_procs)]

        start_time = time.time()

        if DEBUG:
            for proc in processes:
                proc.run()
        else:
            for proc in processes:
                proc.start()

            for proc in processes:
                proc.join()
            print(f"\nTime elapse {time.time() - start_time} sec")

        if args.wandb_track:
            data = wandb.Artifact(
                S3_BUCKET + "-tfrecords",
                type="dataset",
                description="tfrecords of speech command dataset")
            data.add_reference(f"s3://{S3_BUCKET}/{S3_PREFIX}", max_objects=31000)
            run.log_artifact(data)

        for lst in success_dict.values():
            tot_success_dict[train_set] += lst

    if not DEBUG:
        for train_set in sets_categories:
            with open(f"data/{train_set}_tfrecords.csv", 'w') as file: # pylint: disable=unspecified-encoding
                random.shuffle(tot_success_dict[train_set])
                for tfrecord in tot_success_dict[train_set]:
                    tfrecord = re.sub(r'\\', '/', tfrecord)
                    file.write(f'{tfrecord}\n')

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    argparser = argparse.ArgumentParser(
        description='Generate TFrecord formatted input data from a raw speech commands dataset')
    argparser.add_argument(
        '-d',
        '--download',
        type    = int,
        default = 0,
        help    = 'download training data')

    argparser.add_argument(
        '-rb',
        '--reverb_prob',
        type    = float,
        default = 0,
        help    = 'percentage of size for reverb dataset')

    argparser.add_argument(
        '-t',
        '--train_dataset_path',
        default = 'data/train.csv',
        help    = 'path to train data file')

    argparser.add_argument(
        '--test_dataset_path',
        default = 'data/test.csv',
        help    = 'path to test data file')

    argparser.add_argument(
        '-s',
        '--datasize_noise',
        type    = int,
        default = -1, # 45000
        help='How many speech samples per noise')

    argparser.add_argument(
        '-n',
        '--num_procs',
        type    = int,
        default = 8,
        help='How many processor cores to use for execution')

    argparser.add_argument(
        '-w',
        '--wandb_track',
        default = False,
        help    = 'Enable tracking of this run in Weights&Biases')

    argparser.add_argument(
        '--wandb_project',
        type    = str,
        default = 'se',
        help='Weights&Biases project name')

    argparser.add_argument(
        '--wandb_entity',
        type    = str,
        default = 'ambiq',
        help    = 'Weights&Biases entity name')

    main(argparser.parse_args())
