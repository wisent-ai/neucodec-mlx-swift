//
//  ISTFT.swift
//  NeuCodec
//
//  The last step: magnitude and phase to audio. The inverse short-time
//  Fourier transform runs on Accelerate, and the head above it turns the
//  backbone's output into the spectrogram it inverts. Split out of
//  NeuCodecMLXDecoder.swift.
//
//  Licensed under Apache 2.0 (same as NeuCodec)
//

import Foundation
import MLX
import MLXNN
import Accelerate

// MARK: - ISTFT using Accelerate

/// Inverse Short-Time Fourier Transform using Accelerate vDSP.
/// This implementation uses CPU-based FFT for efficiency and to avoid MLX resource limits.
/// Matches PyTorch's torch.fft.irfft behavior for neural vocoder compatibility.
@available(iOS 16.0, *)
struct ISTFT {
    let nFFT: Int
    let hopLength: Int
    let window: [Float]
    let windowSquaredSum: [Float]  // Pre-computed for NOLA normalization

    // DFT setup for real-signal inverse FFT (supports non-power-of-2 sizes like 1920)
    private let dftSetup: OpaquePointer

    init(nFFT: Int = 1920, hopLength: Int = 480, window: [Float]? = nil) {
        self.nFFT = nFFT
        self.hopLength = hopLength

        // Create or use provided window
        if let window {
            self.window = window
        } else {
            // Periodic Hann window (required for perfect reconstruction in STFT/ISTFT)
            // Formula: w[n] = 0.5 * (1 - cos(2π * n / N)) for n = 0..N-1
            // This matches torch.hann_window(N, periodic=True)
            var hannWindow = [Float](repeating: 0, count: nFFT)
            let twoPi = 2.0 * Float.pi
            for i in 0..<nFFT {
                hannWindow[i] = 0.5 * (1.0 - cos(twoPi * Float(i) / Float(nFFT)))
            }
            self.window = hannWindow
        }

        // Pre-compute window squared
        var windowSq = [Float](repeating: 0, count: nFFT)
        vDSP_vsq(self.window, 1, &windowSq, 1, vDSP_Length(nFFT))
        self.windowSquaredSum = windowSq

        // Create DFT setup for real-signal inverse FFT
        // vDSP_DFT_zrop supports arbitrary sizes (not just power-of-2)
        self.dftSetup = vDSP_DFT_zrop_CreateSetup(
            nil,
            vDSP_Length(nFFT),
            .INVERSE
        )!
    }

    /// Perform inverse STFT on magnitude and phase arrays.
    /// - Parameters:
    ///   - magnitude: Float array of shape (numFrames, numBins) flattened row-major
    ///   - phase: Float array matching magnitude shape
    ///   - numFrames: Number of STFT frames
    /// - Returns: Audio waveform as Float array
    func inverse(magnitude: [Float], phase: [Float], numFrames: Int) -> [Float] {
        let numBins = nFFT / 2 + 1
        let outputLength = (numFrames - 1) * hopLength + nFFT

        var output = [Float](repeating: 0, count: outputLength)
        var windowEnvelope = [Float](repeating: 0, count: outputLength)

        // Process each frame
        for frame in 0..<numFrames {
            let frameOffset = frame * numBins

            // Build complex spectrum
            // Input bins 0 to N/2 (961 bins for N=1920)
            var realIn = [Float](repeating: 0, count: nFFT / 2)
            var imagIn = [Float](repeating: 0, count: nFFT / 2)

            // DC component (bin 0) goes to realIn[0]
            // Nyquist component (bin N/2) goes to imagIn[0]
            // This is the packed format expected by vDSP_DFT_zrop
            let dcMag = magnitude[frameOffset]
            let dcPhase = phase[frameOffset]
            realIn[0] = dcMag * cos(dcPhase)

            let nyquistMag = magnitude[frameOffset + nFFT / 2]
            let nyquistPhase = phase[frameOffset + nFFT / 2]
            imagIn[0] = nyquistMag * cos(nyquistPhase)

            // Bins 1 to N/2-1
            for bin in 1..<(nFFT / 2) {
                let mag = magnitude[frameOffset + bin]
                let ph = phase[frameOffset + bin]
                realIn[bin] = mag * cos(ph)
                imagIn[bin] = mag * sin(ph)
            }

            // Perform inverse DFT
            var timeFrame = [Float](repeating: 0, count: nFFT)
            performIFFT(realIn: realIn, imagIn: imagIn, output: &timeFrame)

            // Apply window and overlap-add
            let startSample = frame * hopLength
            for i in 0..<nFFT {
                let idx = startSample + i
                if idx < outputLength {
                    output[idx] += timeFrame[i] * window[i]
                    windowEnvelope[idx] += windowSquaredSum[i]
                }
            }
        }

        // Normalize by window envelope (NOLA condition)
        let eps: Float = 1e-8
        for i in 0..<outputLength {
            if windowEnvelope[i] > eps {
                output[i] /= windowEnvelope[i]
            }
        }

        // Trim padding for "same" mode
        let expectedLength = numFrames * hopLength
        let extraSamples = outputLength - expectedLength

        if extraSamples > 0 {
            let trimStart = extraSamples / 2
            let trimEnd = extraSamples - trimStart
            return Array(output[trimStart..<(outputLength - trimEnd)])
        }
        return output
    }

    /// Perform inverse DFT using vDSP_DFT_zrop (real output).
    /// Input is in packed format where realIn[0]=DC, imagIn[0]=Nyquist.
    private func performIFFT(realIn: [Float], imagIn: [Float], output: inout [Float]) {
        var realInput = realIn
        var imagInput = imagIn

        // Output buffers for the DFT
        // vDSP_DFT_zrop inverse outputs even samples in realOutput, odd samples in imagOutput
        var realOutput = [Float](repeating: 0, count: nFFT / 2)
        var imagOutput = [Float](repeating: 0, count: nFFT / 2)

        realInput.withUnsafeMutableBufferPointer { realInPtr in
            imagInput.withUnsafeMutableBufferPointer { imagInPtr in
                realOutput.withUnsafeMutableBufferPointer { realOutPtr in
                    imagOutput.withUnsafeMutableBufferPointer { imagOutPtr in
                        vDSP_DFT_Execute(
                            dftSetup,
                            realInPtr.baseAddress!,
                            imagInPtr.baseAddress!,
                            realOutPtr.baseAddress!,
                            imagOutPtr.baseAddress!
                        )
                    }
                }
            }
        }

        // Unpack interleaved output: realOutput[k] = output[2k], imagOutput[k] = output[2k+1]
        for i in 0..<(nFFT / 2) {
            output[2 * i] = realOutput[i]
            output[2 * i + 1] = imagOutput[i]
        }

        // Scale by 1/N to match torch.fft.irfft with norm="backward"
        // vDSP_DFT_zrop inverse is not normalized
        var scale = 1.0 / Float(nFFT)
        vDSP_vsmul(output, 1, &scale, &output, 1, vDSP_Length(nFFT))
    }
}

// MARK: - ISTFTHead

/// Head that converts backbone features to audio via ISTFT.
@available(iOS 16.0, *)
final class ISTFTHead: Module {
    let nFFT: Int
    let numBins: Int
    let hopLength: Int
    var istft: ISTFT

    var out: Linear

    init(dim: Int = 1024, nFFT: Int = 1920, hopLength: Int = 480) {
        self.nFFT = nFFT
        self.hopLength = hopLength
        self.numBins = nFFT / 2 + 1  // 961 for nFFT=1920
        self.istft = ISTFT(nFFT: nFFT, hopLength: hopLength)

        // Output: magnitude + phase for each bin
        self.out = Linear(dim, numBins * 2)
    }

    /// Set the ISTFT window from loaded weights.
    func setWindow(_ window: [Float]) {
        istft = ISTFT(nFFT: nFFT, hopLength: hopLength, window: window)
    }

    /// Convert backbone features to audio.
    /// - Parameter x: Backbone output of shape (B, T, D)
    /// - Returns: Audio samples as Float array
    func callAsFunction(_ x: MLXArray) -> [Float] {
        // x: (B, T, D)
        let numFrames = x.dim(1)
        let proj = out(x)  // (B, T, numBins*2)

        // Split into magnitude and phase
        let mag = proj[.ellipsis, ..<numBins]   // (B, T, numBins)
        let phase = proj[.ellipsis, numBins...]  // (B, T, numBins)

        // exp(mag) with clipping for magnitude
        let magExp = minimum(exp(mag), MLXArray(100.0))

        // Evaluate MLX arrays and convert to Float arrays
        // Shape after squeeze: (T, numBins) - keep as frame-major order for ISTFT
        eval(magExp, phase)

        // ISTFT expects frame-major: [frame0_bin0, frame0_bin1, ..., frame1_bin0, ...]
        // DO NOT transpose - asArray flattens row-major which gives us frame-major order
        let magFlat = magExp.squeezed(axis: 0).asArray(Float.self)   // (T, numBins) flattened
        let phaseFlat = phase.squeezed(axis: 0).asArray(Float.self)  // (T, numBins) flattened

        // Call ISTFT with native Float arrays
        return istft.inverse(magnitude: magFlat, phase: phaseFlat, numFrames: numFrames)
    }
}
