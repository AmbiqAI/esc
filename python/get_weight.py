import soundfile as sf
import librosa
import numpy as np

with open("data/train.csv") as file:
    lines = file.readlines()
    
lines = [line.strip() for line in lines]

weights = np.zeros(10)
for i, line in enumerate(lines):
    print(f"\r {i}", end = '')
    fname, d = line.split(',')
    idx = int(d)
    data, samplerate = sf.read(fname)
    if len(data.shape) > 1:
        data = data[:,0]

    if samplerate > 8000:
        out = librosa.resample(data, orig_sr=samplerate, target_sr= 8000)
        weights[idx] += len(out)
print("")
tmp = (1/weights)/np.sum(1/weights)
print(tmp)
    