"""
OFDM transmitter and receiver with energy dispersal, pilot tones and cyclic prefix
"""
#	OFDM transmission and reception with energy dispersal
#
#	Copyright (C) 2020 Bernd Porr <mail@berndporr.me.uk>
#                 2021 David Hutchings <david.hutchings@glasgow.ac.uk>

#	This program is free software; you can redistribute it and/or modify
#	it under the terms of the GNU Lesser General Public License as published by
#	the Free Software Foundation; either version 2 of the License, or
#	(at your option) any later version.

#	This program is distributed in the hope that it will be useful,
#	but WITHOUT ANY WARRANTY; without even the implied warranty of
#	MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#	GNU General Public License for more details.

#	You should have received a copy of the GNU Lesser General Public License
#	along with this program; if not, write to the Free Software
#	Foundation, Inc., 675 Mass Ave, Cambridge, MA 02139, USA.

import numpy as np
import scipy.signal
import komm


def setpilotindex(nData, mQAM, pilotspacing):
    """
    Returns an array of evenly spaced pilots which can then be used for pilotIndices 
    in the OFDM constructor.
    Paramaters nData and mQAM must match the OFDM constructor arguments.
    """
    kpilot = (4*nData//mQAM+pilotspacing//2)//(pilotspacing-1)
    pilotindex = np.arange(pilotspacing//2,kpilot*pilotspacing+pilotspacing//2,pilotspacing,dtype=int)
    pilotindex = np.concatenate((-1*np.flip(pilotindex),pilotindex))
    return pilotindex


class OFDM:
    """
    OFDM encoder and decoder. The data is encoded as QAM using the komm package. 
    Energy dispersal is done with a pre-seeded random number generator. Both pilot 
    tones and the cyclic prefix are added so that the start of the symbol can be 
    detected at the receiver. 
    The complex time series after the inverse Fourier Transform can be modulated 
    into a real valued stream with a Nyquist quadrature modulator for baseband. 
    On the receiver side the start of the symbol is detected by first doing a 
    coarse search with the cyclic prefix and then a precision alignment with the 
    pilots.
    """
    def __init__(self, nFreqSamples = 64, pilotIndices = [-21, -7, 7, 21], 
                 pilotAmplitude = 1, nData = 12, fracCyclic = 0.25, mQAM = 2):
        """
        nFreqSamples sets the number of frequency coefficients of the FFT. Pilot 
        tones are injected at pilotIndices. The real valued pilot amplitude is 
        pilotAmplitude. For transmission nData bytes are expected in an array. 
        The relative length of the Cyclic prefix is fracCyclic. 
        Number of QAM symbols = 2**mQAM, giving mQAM bits per QAM symbol. Average 
        power is normalised to unity. Default example correspond to 802.11a wifi 
        modulation.
        """
        # the total number of frequency samples
        self.nIFFT = nFreqSamples

        # number of data samples in bytes (coded/decoded)
        self.nData = nData

        # fracCyclic is relative cyclic prefix/ guard interval length
        self.nCyclic = int(self.nIFFT * fracCyclic)

        # indices of pilots	
        self.pilotIndices = np.array(pilotIndices)
        
        # amplitudes of the pilot carrier at the beginning
        self.pilotAmplitude = pilotAmplitude
        
        # mQAM bits per QAM symbol
        self.mQAM = mQAM
        # normalisation to make average bit energy unity for square QAM
        norm = 0
        for i in range(1,2**(mQAM//2),2):
            norm = norm + i**2
        self.norm = np.sqrt(norm*2**(2-mQAM//2))
        
        # use komm open-source library for QAM mod/demod 
        # pypi.org/project/komm 
        # by Roberto W. Nobrega <rwnobrega@gmail.com>
        self.constellation = komm.QAMConstellation(2**self.mQAM, deltas=1./self.norm)  # 16 symbols
        self.labeling = komm.ReflectedRectangularLabeling(self.mQAM)  # 4 bits per symbol

        self.kstart = (8*self.nData//self.mQAM+self.pilotIndices.size)//2

    def modulate(self, bits):
        indices = self.labeling.bits_to_indices(bits)
        return self.constellation.indices_to_symbols(indices)

    def demodulate_hard(self, received):
        indices_hat = self.constellation.closest_indices(received)
        return self.labeling.indices_to_bits(indices_hat)

    def encode(self,data,randomSeed = 1):
            
        """
        Creates an OFDM symbol using QAM. 
        The signal is a complex valued numpy array where the
        encoded data-stream is appended. 
        The data is an array of bytes.
        The random seed sets the pseudo random number generator for the
        energy dispersal.
        """
       
        # set the random number generator to a known start value
        # will generate always the same sequence from this start value
        # We xor its value with the grey values from the image to
        # generate a pseudo random sequence which is called "energy dispersal".
        rng = np.random.default_rng(randomSeed)
        rints = np.uint8(rng.integers(256,size=self.nData))
        data = data ^ rints
        bin_data = np.unpackbits(data).flatten()
        tx_data = self.modulate(bin_data)
      
        # create an empty spectrum with all complex frequency values set to zero
        self.spectrum = np.zeros(self.nIFFT,dtype=complex)
        
        idata = 0
        ipilot = 0
        for k in range(-self.kstart,0):
            if ipilot==self.pilotIndices.size or k!=self.pilotIndices[ipilot]:
                self.spectrum[k] = tx_data[idata]
                idata+=1
            else:               
                self.spectrum[k] = self.pilotAmplitude  
                ipilot+=1
                
        for k in range(1,self.kstart+1):
            if ipilot==self.pilotIndices.size or k!=self.pilotIndices[ipilot]:
                self.spectrum[k] = tx_data[idata]
                idata+=1
            else:               
                self.spectrum[k] = self.pilotAmplitude  
                ipilot+=1       
            
        # Create one symbol by transforming our frequency samples into
        # complex timedomain samples
        complex_symbol = np.fft.ifft(self.spectrum)

        # Generate cyclic prefix taken from the end of the signal
        cyclicPrefix = complex_symbol[-self.nCyclic:]

        # Add the cyclic prefix to the complex valued symbol 
        complex_symbol = np.append(cyclicPrefix,complex_symbol)
        return complex_symbol

    def initDecode(self,signal,offset):
        """
        Starts a decoding process. The signal is the real valued received
        signal and the decoding start at the index specified by offset.
        """
        self.rxindex = offset
        self.signal = signal

    def decode(self, randomSeed = 1):
        """
        Decodes one symbol and returns a byte array of the
        data and the sum of the squares of the imaginary parts
        of the pilot tones. The smaller that value the better
        the symbol start detection, the reception and the jitter 
        (theoretically zero at perfect reception).
        """     
        
        # Skip cyclic prefix
        self.rxindex = self.rxindex + self.nCyclic
        rx_symbol = self.signal[self.rxindex:self.rxindex+self.nIFFT]

        # Perform an FFT to get the frequency samples which code our signal as QPSK pairs
        rx_freqs = np.fft.fft(rx_symbol)
        
        # sum of the square of the imaginary parts of the pilot tones
        imPilots = 0.0
        
        # the byte array storing the received data
        rx_data = np.zeros(8*self.nData//self.mQAM,dtype=complex)
        
        idata = 0
        ipilot = 0
        for k in range(-self.kstart,0):
            if ipilot==self.pilotIndices.size or k!=self.pilotIndices[ipilot]:
                rx_data[idata] = rx_freqs[k] 
                idata+=1
            else:               
                imPilots += np.imag(rx_freqs[k])**2
                ipilot+=1
                
        for k in range(1,self.kstart+1):
            if ipilot==self.pilotIndices.size or k!=self.pilotIndices[ipilot]:
                rx_data[idata] = rx_freqs[k] 
                idata+=1
            else:               
                imPilots += np.imag(rx_freqs[k])**2 
                ipilot+=1  
                
        rx_bin = self.demodulate_hard(rx_data)
       
        # now let's assemble the bits into into bytes
        rx_byte = np.packbits(rx_bin)

        # set the random number generator to the same value as in the transmitter so that
        # we have exactly the same sequence
        rng = np.random.default_rng(randomSeed)
        # de-scramble the bytes
        rints = np.uint8(rng.integers(256,size=self.nData))
        rx_byte = rx_byte ^ rints

        # increment rxindex for next symbol        
        self.rxindex = self.rxindex+self.nIFFT

        return rx_byte, imPilots

    def findSymbolStartIndex(self, signal, searchrangecoarse=None, searchrangefine = 25):
        """
        Finds the start of the symbol by 1st doing a cross correlation @nIFFT
        with the cyclic prefix and then it uses the pilot tones.
        Arguments: the real valued reception signal, the coarse searchrange for
        the cyclic prefix and the fine one for the pilots.
        Returns the cross correlation value array from the cyclic prefix,
        the squared values of the imaginary parts of the pilots and the 
        index of the symbol start relative to the signal.
        """
        # Set it to some default
        if not searchrangecoarse:
            searchrangecoarse = self.nIFFT*3
            
        # Let find the starting index with the cyclic prefix
        crosscorr = np.array([])
        for i in range(searchrangecoarse):
            s1 = signal[i:i+self.nCyclic]
            s2 = signal[i+self.nIFFT:i+self.nIFFT+self.nCyclic]
            crosscorr = np.append(crosscorr, np.real(np.sum(s1*np.conj(s2))))
            
        pks,_ = scipy.signal.find_peaks(crosscorr,distance=self.nIFFT,width=self.nCyclic//2)
        o1 = pks[0]
              
        # Now let's fine tune it by looking at the imaginary parts
        imagpilots = np.array([])
        for i in range(o1-searchrangefine,o1+searchrangefine):
            self.initDecode(signal,i)
            im = self.decode()[1]
            imagpilots = np.append(imagpilots,im)

        # Correct it with the pilots
        o2 = o1 + np.argmin(imagpilots) - searchrangefine
        return crosscorr,imagpilots,o2
