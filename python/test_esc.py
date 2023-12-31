"""
Test trained NN model using wavefile as input
"""
import argparse
import os
import re
import wave
import numpy as np
import scipy
import soundfile as sf
import sounddevice as sd
import librosa
from nnsp_pack.feature_module import display_stft_tfmask_esc
from nnsp_pack.pyaudio_animation import AudioShowClass
from nnsp_pack.nn_infer import NNInferClass
from nnsp_pack.stft_module import stft_class
from nnsp_pack.basic_dsp import dc_remove
from data_esc_nonoverlap import params_audio as PARAM_AUDIO

SHOW_HISTOGRAM  = False
NP_INFERENCE    = False

CLASSSES_SET=[
    "unknown",
    "dog_bark",
    "gun_shot",
    "siren_car_horn",
    "baby_crying"]

class SeClass(NNInferClass):
    """
    Class to handle SE model
    """
    def __init__(
            self,
            nn_arch,
            epoch_loaded,
            params_audio,
            quantized=False,
            show_histogram=False,
            np_inference=False,
            feat_type = 'mel'):

        super().__init__(
            nn_arch,
            epoch_loaded,
            params_audio,
            quantized,
            show_histogram,
            np_inference,
            feat_type = feat_type)

        self.fbank_mel = np.load('fbank_mel.npy')

    def reset(self):
        """
        Reset se instance
        """
        print("Reset all the states from parents")
        super().reset()

    def blk_proc(self, data, wavefile="speech.wav"):
        """
        NN process for several frames
        """
        result_folder = 'test_results'
        os.makedirs(result_folder, exist_ok=True)
        _, fname = os.path.split(wavefile)
        folder = re.sub(r'\.wav', '', fname)
        result_folder = f"{result_folder}/{folder}"
        os.makedirs(result_folder, exist_ok=True)
        params_audio = self.params_audio
        file = wave.open(f"{result_folder}/output.wav", "wb")
        file.setnchannels(2)
        file.setsampwidth(2)
        file.setframerate(params_audio['sample_rate'])

        bks = int(len(data) / params_audio['hop'])
        feats   = []
        specs   = []
        tfmasks = []
        stft_inst = stft_class(
            hop=params_audio['hop'],
            fftsize=params_audio['len_fft'],
            winsize=params_audio['win_size'])
        stft_inst.reset()
        data_frame = np.ones((params_audio['hop'],),dtype=np.float64) * 2**-15
        data_freq = stft_inst.stft_frame_proc(data_frame)
        data_freq_stack = [data_freq.copy() for i in range(self.len_filter)]
        for i in range(bks):
            data_frame = data[i*params_audio['hop'] : (i+1) * params_audio['hop']]
            data_freq = stft_inst.stft_frame_proc(data_frame)
            data_freq_stack.pop(0)
            data_freq_stack += [data_freq]
            if NP_INFERENCE:
                feat, spec, est = self.frame_proc_np(data_frame, return_all = True)
            else:
                feat, spec, est = self.frame_proc_tf(data_frame, return_all = True)
            est = scipy.special.softmax(est)
            print(f": {CLASSSES_SET[np.argmax(est)]}: {est.max():.2f}")
            out = np.ones((params_audio['hop'],)) * float(np.argmax(est)) /5
            tfmasks += [est]
            feats   += [feat]
            specs   += [spec]
            self.count_run = (self.count_run + 1) % self.num_dnsampl
            print(f"\rprocessing frame {i}", end='')
            out = np.array([data_frame, out]).T.flatten()
            out = np.floor(out * 2**15).astype(np.int16)
            file.writeframes(out.tobytes())

        print('\n', end='')
        tfmasks = np.array(tfmasks)
        feats   = np.array(feats)
        specs   = np.array(specs)

        display_stft_tfmask_esc(
            data,
            specs.T,
            feats.T,
            tfmasks.T,
            sample_rate=params_audio['sample_rate'],
            print_name=f"{result_folder}/output.pdf",
            mask_label=CLASSSES_SET)

        file.close()

        data, samplerate = sf.read(f"{result_folder}/output.wav")
        savewav = f"{result_folder}/est.wav"
        sf.write(f"{savewav}", data[:,1], samplerate)
        print(f'Check your environmental sound classification in {savewav}')

def main(args):
    """main function"""
    epoch_loaded    = int(args.epoch_loaded)
    quantized       = args.quantized
    recording       = int(args.recording)
    test_wavefile   = args.test_wavefile

    if recording == 1:
        wavefile='test_wavs/speech.wav'
        AudioShowClass(
                record_seconds=10,
                wave_output_filename=wavefile,
                non_stop=False)
    else:
        wavefile = test_wavefile

    data, sample_rate = sf.read(wavefile)
    if data.ndim > 1:
        data=data[:,0]

    if sample_rate > PARAM_AUDIO['sample_rate']:
        data = librosa.resample(
                data,
                orig_sr=sample_rate,
                target_sr=PARAM_AUDIO['sample_rate'])

    # data = dc_remove(data)
    sd.play(data, PARAM_AUDIO['sample_rate'])

    se_inst = SeClass(
            args.nn_arch,
            epoch_loaded,
            PARAM_AUDIO,
            quantized,
            show_histogram  = SHOW_HISTOGRAM,
            np_inference    = NP_INFERENCE,
            feat_type       = args.feat_type,
            )

    se_inst.blk_proc(data, wavefile=wavefile)

if __name__ == "__main__":

    argparser = argparse.ArgumentParser(
        description='Testing trained SE model')

    argparser.add_argument(
        '-a',
        '--nn_arch',
        default='nn_arch/def_esc_nn_arch.txt',
        help='nn architecture')

    argparser.add_argument(
        '-ft',
        '--feat_type',
        default='mel',
        help='feature type: \'mel\'or \'pspec\'')

    argparser.add_argument(
        '-r',
        '--recording',
        default = 0,
        help    = '1: recording the speech and test it, \
                   0: No recording.')

    argparser.add_argument(
        '-v',
        '--test_wavefile',
        default = 'test_wavs/gun_shot.wav',
        help    = 'The wavfile name to be tested')
    # ["unknown", "dog_bark", "gun_shot", "siren", "car_horn", "baby_crying"]
    argparser.add_argument(
        '-q',
        '--quantized',
        default = False,
        type=bool,
        help='is post quantization?')

    argparser.add_argument(
        '--epoch_loaded',
        default=2250, # 70
        help='starting epoch')

    main(argparser.parse_args())
