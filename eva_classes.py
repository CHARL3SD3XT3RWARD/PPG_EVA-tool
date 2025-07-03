import numpy as np
import scipy as sc
import scipy.signal as scsignal
from scipy.stats import entropy, skew, kurtosis
from scipy.interpolate import make_interp_spline as C2_Spline #C2 Continuity
from scipy.interpolate import PchipInterpolator as pchip
import pandas as pd

import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter

import eva_toolkit as kit

import configparser

config = configparser.ConfigParser()
config_path = r''

config.read(config_path) # path to config.ini

<<<<<<< HEAD
config.read(r'') # path to config.ini

=======
>>>>>>> d479f2cdcb398b3eacaa157d91ce9075a2f1abe0
fs_A = int(config['Settings']['fs_a'])
low_BPM = 48
high_BPM = 120 #workaround for deprecated funktion


class Processing:
    '''
    The class for processing the Signals. It takes the signal as an array and the timestamps as a pandas.Series objekt. Included are:\n
        pleth_filter\n
        resampling\n
        slice_list\n
    
    '''
    def __init__(self, signals, timestamps):
        '''
        

        Parameters
        ----------
        signals : array
            The signal.
        timestamps : Series
            The coresponding timestamps as Pandas.Series Objekt.

        Returns
        -------
        None.

        '''
        self.signals = signals
        self.timestamps = timestamps
        
    def pleth_filter(self, fs, lowcut, highcut, order, resampled=False):
        '''
        Butterwoth IIR filter. It takes the signal and filters it in a second order section mannor. 
        If the signal was resampled with Processing.resampled, resampled must be set True.
        The filtered signal is stored in self.filtered_signal.

        Parameters
        ----------
        fs_A : Int
            Sampling rate in Hertz. The samplingrate is provided with the config.ini. For more information on that go to PPG_EVA_GUI.set_values().
        lowcut : float, optional
            The lower bound of the Filter. It is provided with the config.ini.
        highcut : float, optional
            The higher bound of the Filter. It is provided with the config.ini.
        order : Int, optional
            The order of the Filter. It is provided with the config.ini. Keep in mind that due to forward-backwards filtering the effektive order doubles.
        resampled : Bool, optional
            Wether the Signal is resampled or not. the resampled signal is a different attribute of this object.
            The default is False.

        '''
        
        sos = scsignal.butter(order, [lowcut, highcut], fs=fs, btype='band', output='sos')
        if resampled:
            self.filtered_signal = scsignal.sosfiltfilt(sos, self.resampled_signal)*(-1)#inverting
            #entfernen des Gleichspannungsanteil
            self.filtered_signal=self.filtered_signal-np.mean(self.filtered_signal)     
        
        else:
            self.filtered_signal = scsignal.sosfiltfilt(sos, self.signals)
            #entfernen des Gleichspannungsanteil
            self.filtered_signal=self.filtered_signal-np.mean(self.filtered_signal)     
    
    
    def resampling(self,  sampling_rate, target_rate=fs_A, interpolate_pchip=False):
        '''
        A function to resample a signal. It can be resampled with the C2-spline-interpolation method or the pchip method.
        This funktion is essentially deprecated but can still be used by modifying the PPG_EVA_tool.preprocessing funktion.
        The resampled signal and timestamps is stored in the self.resample_signal and self.new_timestamps atributes respectivley.
        
        Parameters
        ----------
        sampling_rate : Int
            The original smaplingrate of the signal.
        target_rate: Int
            The target samplingrate of the signal. This parameter is provided by the config.ini as fs_A.
        interpolate_pchip: bool
            if True, the resampling is performed by the pchip-method. 
            The default is False and hence C2-Splineinterpolation is used.

        '''
        
        # number of samples in original signal
        n_orig = len(self.signals)
        
        # target number of samples
        n_target = int(n_orig * (target_rate / sampling_rate))
        
        # create old and new timeaxis evenly
        t_old = np.linspace(0, 1, n_orig, endpoint=False)
        t_new = np.linspace(0, 1, n_target, endpoint=False)
        
        if interpolate_pchip:
            splineC1=pchip(t_old, self.signals)
            signal_newC1=splineC1(t_new)
            self.resampled_signal=signal_newC1(t_new)
            self.new_timestamps=pd.to_datetime(t_new, unit='s')
    
        else:
            spline=C2_Spline(t_old, self.signals, k=3)
            self.resampled_signal=spline(t_new)
            self.new_timestamps=pd.to_datetime(t_new, unit='s')

        
    def slice_list(self, chunk_size):
        '''
        Sequencing of the signal into desired length. Before storing the signalchunks it is checked that all chunks are of the same length.
        The chunks are stored in the self.signal_chunks atribute as a 2D-array.

        Parameters
        ----------
        chunk_size : Int, optional
            The desired length of sequence.

        '''
        self.filtered_signal = self.filtered_signal[512:]#cut away the settlement time of the filter
        
        signal_chunks=[]
        for i in range(0, len(self.filtered_signal), chunk_size):
            chunk = self.filtered_signal[i:i+chunk_size]
            if len(chunk) == chunk_size:
                signal_chunks.append(chunk)
        
        self.signal_chunks = signal_chunks    
                    
class SQI:
    '''

    The class to calculate all SQIs. To initialize it takes only a 2D-array of the signalchunks. Includes:´\n
    shanon_entropy\n
    calc_SNR\n
    ZCR\n
    skewness\n
    kurt\n
    
    '''
    
    def __init__(self, signal_chunks):
        '''
        

        Parameters
        ----------
        signal_chunks : array
            A 2D-array with the sequences.
 

        '''

        self.signal_chunks = signal_chunks

        
    def shanon_entropy(self, num_bins=16):
        '''
        Calculates the normalised entropy for every sequence.
        The entropy for every chunk is stored in  a array in the self.entropy_values atribute.

        Parameters
        ----------
        num_bins : Int, optional
            The number of bins used to create the histogram and therefore the probability density function.
            The default is 16.

        '''
        entropy_values=[]
        for a in range(len(self.signal_chunks)):
            quantized_signal, bin_edges = np.histogram(self.signal_chunks[a], bins=num_bins, density=True)
            probabilities = quantized_signal / np.sum(quantized_signal)
            # Shannon-Entropie berechnen
            shannon_entropy = entropy(probabilities)/-np.log(1/num_bins)#normalised entropy by the value of equal probabilities
            entropy_values=np.append(entropy_values, shannon_entropy)
            
        self.entropy_values = entropy_values
        
    def calc_SNR(self, lowcut=low_BPM/60, highcut=high_BPM/60):
        '''
        Calculates the Signal- to Noise-Ratio.
        Defined as the relative power of a signalband to the rest. 
        To determine the signalpower, the FFT of the given chunk is integrated within the lowcut and highcut as lower and upper limits respectivley.
        The noisepower is the integrates value of the FFT outside these boundaries.

        Parameters
        ----------
        lowcut : float, optional
            Lower bound of the frequency band. This parameter is provided by the config.ini as bpm.
        highcut : float, optional
            higher bound of the frequency band. This parameter is provided by the config.ini as bpm.

        Returns
        -------
        self.SNR: list
            The SNR-values for every sequence.
        self.freq: list
            the frequencies for every sequence.
        self.magnitude: list
            the magnitudes of the frequencies for every sequence.

        '''
        snr=[]
        frequencies = []
        magnitudes = []
        for i in range(len(self.signal_chunks)):
            freq = np.fft.fftfreq(len(self.signal_chunks[i]), d=1/fs_A)
            fft_values=np.fft.fft(self.signal_chunks[i])
            magnitude = np.abs(fft_values)
    
            signal_band = (freq >= lowcut) & (freq <= highcut)
            # signal_power = np.mean(np.sum(fft_values[signal_band])**2)
            signal_power = np.abs(np.trapz(fft_values[signal_band]))
            
            
            # noise_band = ~signal_band  # Frequenzbereich außerhalb der Signalbandbreite
            noise_band = (freq > highcut)  # Frequenzbereich außerhalb der Signalbandbreite
            # noise_band = (freq < lowcut)  # Frequenzbereich außerhalb der Signalbandbreite
            # noise_power = np.mean(np.sum(fft_values[noise_band])**2)
            noise_power = np.abs(np.trapz(fft_values[noise_band]))
            
            snr_db = 10 * np.log10(signal_power / noise_power)
            snr.append(snr_db)
            frequencies.append(freq)
            magnitudes.append(magnitude)
            
        self.SNR = (snr)
        self.freq = frequencies
        self.magnitude = magnitudes
            
    def ZCR(self):
        '''
        Calculate the ZCR for every sequence. The ZCR is here defined as the variance to a lin. Regression of the time of occurance of every signchange.
        The lin. Regression is performed with the eva_toolkit.lin_reg() function and the variance is calculated by the eva_toolkit.variance() function.

        Returns
        -------
        self.signs: list
            A list with values for the signs of the signalvalues.\n
            1 = positive\n
            -1 = negative
        self.cross_pos: list
            A list of the indices where a zero crossing occured.
        self.n_cross: Int
            The number of zero crossings.
        self.slope: float
            the slope of the lin. regression for every sequence.
        self.intersect: float
            the y-intersect of the lin. regression for every sequence.
        self.variance:
            the variance for every sequence.
            
        '''
        
        signs=[]
        cross_pos=[]
        n_cross=[]
        for i in range(len(self.signal_chunks)):
            tempsigns=np.sign(self.signal_chunks[i])
            crossings=np.diff(tempsigns)
            tempcrossing_positions=np.where(crossings != 0)[0] #only True where crossing[i] != crossing[i+1]
            temp_n_crossings=len(tempcrossing_positions)    
            
            signs.append(tempsigns)
            cross_pos.append(tempcrossing_positions)
            n_cross.append(temp_n_crossings)
        
        self.signs = signs
        self.cross_pos = cross_pos
        self.n_cross = n_cross

        self.slope, self.intersect = kit.lin_reg(self.signal_chunks, cross_pos)
        self.variance = kit.variance(self.slope, self.intersect, cross_pos)

    def skewness(self):
        '''
        Calculates the skewness for every sequence.

        Returns
        -------
        self.skewness: list
            The skewness for every sequence.

        '''
        
        skewness = []
        
        for chunk in self.signal_chunks:
            skewness.append(skew(chunk))
            
        self.skewness = skewness
        
    def kurt(self):
        '''
        Calculates the kurtosis for every sequence.

        Returns
        -------
        self.kurt: list
            The kurtosis for every sequence.

        '''

        
        kurt = []
        
        for chunk in self.signal_chunks:
            kurt.append(kurtosis(chunk))
            
        self.kurt = kurt
        
class Training:
    '''
    The class for Training. It takes the skewness, and kurtosis as separate arrays and the anotation as a list or array as a binary classification.\n
    Includes:\n
        separate_values\n
        building hists\n
    
    '''
    def __init__(self, skew_values, kurt_values, data):
        '''
        

        Parameters
        ----------
        skew_values : array
            The array with all skewness values.
        kurt_values : array
            The array with all kurtosis values.
        data : array
            The annotation.


        '''
        
        self.skew_values = np.asarray(skew_values)
        self.kurt_values = np.asarray(kurt_values)
        self.data = data
        
    def separate_values(self):
        '''
        Separating the values in good and bad data according to the annotation.
        
        Returns
        -------
        self.positive_skew: array
            All skewness values that are labeled as good.
        self.positive_kurt: array
            All kurtosis values that are labeled as good.
        self.negative_skew: array
            All skewness values that are labeled as bad.
        self.negative_kurt: array
            All kurtosis values that are labeled as bad.
        
        '''
        
        if len(self.data)-len(self.skew_values) == 0: #check for equal length -> human error
            positive_idx = np.where(np.asarray(self.data) == 1)[0]
            negative_idx = np.where(np.asarray(self.data) == 0)[0]
    
            #sorting the values with respect to the annotation
            self.positive_skew = self.skew_values[positive_idx]
            self.positive_kurt = self.kurt_values[positive_idx]
            
            self.negative_skew = self.skew_values[negative_idx]
            self.negative_kurt = self.kurt_values[negative_idx]
        
        else:
            print('Annotation and Signal are not of the same lenght!')
            print(f'Annotation {len(self.data)}, Signal {len(self.skew_values)}')
            
    def building_hists(self, plot=False):
        '''
        Builds a 2D probability densitifunction where the skewness and the kurtosis the two dimensions resemble.
        The resulting PDF is a logarithmic difference of the PDFs of the good data and the bad data.        

        Parameters
        ----------
        plot : bool, optional
            If True, all PDFs will be plotted. The default is False.

        Returns
        -------
        self.hist: aray
            The PDF-values.
        self.xedges: array
            The xedges of the bins.
        self.yedges: array            
            The yedges of the bins.

        '''
            
        bins = 1000
        range_xy=[[-3, 3], [-3, 3]]
        
        hist_good, xedges, yedges = np.histogram2d(self.positive_skew, self.positive_kurt, bins=bins, range=range_xy, density=True)
        hist_bad, _, _= np.histogram2d(self.negative_skew, self.negative_kurt, bins=bins, range=range_xy, density=True)
        
        #normalising hists
        hist_good = hist_good/hist_good.sum()
        hist_bad = hist_bad/hist_bad.sum()
        
        hist_diff = np.log(hist_good + 1e-10) - np.log(hist_bad + 1e-10)
      
        hist_smoothed = gaussian_filter(hist_diff, sigma=1.5)  # sigma kannst du anpassen
        
        self.hist = hist_smoothed
        self.xedges = xedges
        self.yedges = yedges
        
        if plot:
            
            cmap = 'afmhot'
            fig, axes= plt.subplots(ncols=2, nrows=2, figsize=(18,18))
            ax1, ax2, ax3, ax4 = axes.ravel() 
            
            extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
            im = ax1.imshow(hist_good.T, origin='lower', extent=extent, aspect='auto', cmap=cmap)
            cbar1 = plt.colorbar(im, ax=ax1, label='Dichte gut')
            cbar1.set_label('Dichte gut')  
            ax1.set_xlabel('Skewness')
            ax1.set_ylabel('Kurtosis')
            
            im = ax2.imshow(hist_bad.T, origin='lower', extent=extent, aspect='auto', cmap=cmap)
            cbar2 = plt.colorbar(im, ax=ax2, label='Dichte schlecht')
            cbar2.set_label('Dichte schlecht')  
            ax2.set_xlabel('Skewness')
            ax2.set_ylabel('Kurtosis')
           
        
            im = ax3.imshow(hist_diff.T, origin='lower', extent=extent, aspect='auto', cmap=cmap)
            cbar3 = plt.colorbar(im, ax=ax3, label='log-Dichtdifferenz')
            cbar3.set_label('log-Dichtedifferenz')
            ax3.set_xlabel('Skewness')
            ax3.set_ylabel('Kurtosis')
            
            im = ax4.imshow(hist_smoothed.T, origin='lower', extent=extent, aspect='auto', cmap=cmap)
            cbar4 = plt.colorbar(im, ax=ax4)
            cbar4.set_label('log-Dichtedifferenz geglättet')
            ax4.set_xlabel('Skewness')
            ax4.set_ylabel('Kurtosis')
            
            ax1.grid(False)
            ax2.grid(False)
            ax3.grid(False)
            ax4.grid(False)
            
            plt.tight_layout()
            plt.show()
