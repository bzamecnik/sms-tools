# matplotlib without any blocking GUI
import matplotlib as mpl

mpl.use('Agg')
import essentia
import essentia.standard as ess
from pylab import *
from numpy import *

from smst.utils import audio, synth
from smst.models import stft

filename = '../../../sounds/carnatic.wav'
hopSize = 128
frameSize = 2048
sampleRate = 44100
guessUnvoiced = True

run_windowing = ess.Windowing(type='hann', zeroPadding=3 * frameSize)  # Hann window with x4 zero padding
run_spectrum = ess.Spectrum(size=frameSize * 4)
run_spectral_peaks = ess.SpectralPeaks(minFrequency=50,
                                       maxFrequency=10000,
                                       maxPeaks=100,
                                       sampleRate=sampleRate,
                                       magnitudeThreshold=0,
                                       orderBy="magnitude")
run_pitch_salience_function = ess.PitchSalienceFunction(magnitudeThreshold=30)
run_pitch_salience_function_peaks = ess.PitchSalienceFunctionPeaks(minFrequency=100, maxFrequency=300)
run_pitch_contours = ess.PitchContours(hopSize=hopSize, peakFrameThreshold=0.8)
run_pitch_contours_melody = ess.PitchContoursMelody(guessUnvoiced=guessUnvoiced,
                                                    hopSize=hopSize)

pool = essentia.Pool();

audio = ess.MonoLoader(filename=filename)()
audio = ess.EqualLoudness()(audio)

for frame in ess.FrameGenerator(audio, frameSize=frameSize, hopSize=hopSize):
    frame = run_windowing(frame)
    spectrum = run_spectrum(frame)
    peak_frequencies, peak_magnitudes = run_spectral_peaks(spectrum)

    salience = run_pitch_salience_function(peak_frequencies, peak_magnitudes)
    salience_peaks_bins, salience_peaks_saliences = run_pitch_salience_function_peaks(salience)

    pool.add('allframes_salience_peaks_bins', salience_peaks_bins)
    pool.add('allframes_salience_peaks_saliences', salience_peaks_saliences)

contours_bins, contours_saliences, contours_start_times, duration = run_pitch_contours(
    pool['allframes_salience_peaks_bins'],
    pool['allframes_salience_peaks_saliences'])
pitch, confidence = run_pitch_contours_melody(contours_bins,
                                              contours_saliences,
                                              contours_start_times,
                                              duration)

yf0 = synth.synthesize_sinusoid(pitch, .6, hopSize, sampleRate)

figure(1, figsize=(9, 6))

mX, pX = stft.from_audio(audio, hamming(frameSize), frameSize, hopSize)
maxplotfreq = 3000.0
numFrames = int(mX.shape[0])
frmTime = hopSize * arange(numFrames) / float(sampleRate)
binFreq = sampleRate * arange(frameSize * maxplotfreq / sampleRate) / frameSize
plt.pcolormesh(frmTime, binFreq, np.transpose(mX[:, :frameSize * maxplotfreq / sampleRate + 1]))
plt.autoscale(tight=True)

offset = .5 * frameSize / sampleRate
time = hopSize * arange(size(pitch)) / float(sampleRate)
pitch[pitch == 0] = nan
plot(time, pitch, color='k', linewidth=2)

plt.title('mX + prominent melody (carnatic.wav)')
tight_layout()
savefig('predominantmelody-2.png')
audio.write_wav(yf0, sampleRate, 'predominantmelody-2.wav')
