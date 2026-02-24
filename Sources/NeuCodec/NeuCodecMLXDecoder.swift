//
//  NeuCodecMLXDecoder.swift
//  NeuCodec
//
//  MLX-based NeuCodec speech decoder.
//  Converts speech codes to 24kHz audio samples using GPU acceleration.
//
//  Architecture:
//  1. FSQ Decode: codes → embeddings (2048 dim)
//  2. Linear projection: 2048 → 1024
//  3. VocosBackbone: Conv1d + ResNet + 12x Transformer + ResNet
//  4. ISTFTHead: Linear → magnitude/phase → ISTFT → audio
//
//  Licensed under Apache 2.0 (same as NeuCodec)
//

import Foundation
import MLX
import MLXNN
import MLXRandom
import Hub
import Accelerate
import os

// MARK: - FSQ Decoder

/// Finite Scalar Quantization decoder.
/// Converts integer codes to continuous embeddings.
@available(iOS 16.0, *)
final class FSQDecoder: Module, UnaryLayer {
    let levels: [Int] = [4, 4, 4, 4, 4, 4, 4, 4]
    let basis: [Int32]

    var projectIn: Linear
    var projectOut: Linear

    override init() {
        // Compute basis for index decomposition: [1, 4, 16, 64, 256, 1024, 4096, 16384]
        var b: [Int32] = [1]
        for i in 0..<7 {
            b.append(b[i] * Int32(levels[i]))
        }
        self.basis = b

        self.projectIn = Linear(2048, 8)
        self.projectOut = Linear(8, 2048)
    }

    /// Decode integer codes to continuous embeddings.
    /// - Parameter indices: Integer codes of shape (B, T)
    /// - Returns: Embeddings of shape (B, T, 2048)
    func callAsFunction(_ indices: MLXArray) -> MLXArray {
        // Expand indices for broadcasting: (B, T) -> (B, T, 1)
        let expanded = indices.expandedDimensions(axis: -1)

        // Convert to level indices: (B, T, 8)
        // IMPORTANT: Use floorDivide for integer division (like Python's //)
        // Regular / does float division which gives wrong results!
        let basisArr = MLXArray(basis).reshaped([1, 1, 8])
        let levelsArr = MLXArray(levels.map { Int32($0) }).reshaped([1, 1, 8])

        let levelIndices = floorDivide(expanded, basisArr) % levelsArr

        // Scale to [-1, 1] range using correct FSQ formula:
        // bound = level_indices / half_width - 1  (where half_width = levels // 2)
        // This maps: 0 -> -1.0, 1 -> -0.5, 2 -> 0.0, 3 -> 0.5 for levels=4
        let halfWidth = levelsArr.asType(.float32) / 2.0
        let scaled = levelIndices.asType(.float32) / halfWidth - 1.0

        // Project to embedding space
        return projectOut(scaled)
    }
}

// MARK: - ResNet Block

/// Residual block with GroupNorm and Swish activation.
/// Uses channels-last format (B, T, C) for MLX compatibility.
@available(iOS 16.0, *)
final class NeuCodecResnetBlock: Module, UnaryLayer {
    var norm1: GroupNorm
    var conv1: Conv1d
    var norm2: GroupNorm
    var conv2: Conv1d

    override init() {
        let channels = 1024
        let groups = 32
        self.norm1 = GroupNorm(
            groupCount: groups, dimensions: channels, eps: 1e-6, affine: true, pytorchCompatible: true
        )
        self.conv1 = Conv1d(
            inputChannels: channels, outputChannels: channels, kernelSize: 3, padding: 1
        )
        self.norm2 = GroupNorm(
            groupCount: groups, dimensions: channels, eps: 1e-6, affine: true, pytorchCompatible: true
        )
        self.conv2 = Conv1d(
            inputChannels: channels, outputChannels: channels, kernelSize: 3, padding: 1
        )
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var h = norm1(x)
        h = silu(h)
        h = conv1(h)
        h = norm2(h)
        h = silu(h)
        h = conv2(h)
        return x + h
    }
}

// MARK: - RoPE (Rotary Position Embeddings)

/// Rotary Position Embeddings applied to attention Q and K.
/// Matches torchtune's RotaryPositionalEmbeddings which uses interleaved pairs.
@available(iOS 16.0, *)
struct RoPE {
    let dim: Int
    let base: Float

    init(dim: Int = 64, base: Float = 10000.0) {
        self.dim = dim
        self.base = base
    }

    /// Apply rotary embeddings to input tensor.
    /// - Parameters:
    ///   - x: Input tensor of shape (B, H, T, D)
    ///   - offset: Position offset for KV cache (not used for decoder)
    /// - Returns: Tensor with rotary embeddings applied
    func callAsFunction(_ x: MLXArray, offset: Int = 0) -> MLXArray {
          // Match Python's behavior: use dim 1 (H) as position index
          let numHeads = x.dim(1)  // H = 16
          let seqLen = x.dim(2)    // T (not used for position)
          let headDim = x.dim(3)

          // Positions based on HEAD INDEX (0 to H-1), not time position
          let positions = MLXArray(Int32(0)..<Int32(numHeads)).asType(.float32)

          // Compute angles for H positions
          let halfDim = dim / 2
          let freqIndices = MLXArray(stride(from: 0, to: dim, by: 2).prefix(halfDim).map { Float($0) })
          let invFreq = 1.0 / pow(MLXArray(base), freqIndices / Float(dim))
          let angles = positions.expandedDimensions(axis: 1) * invFreq.expandedDimensions(axis: 0)

          let cosCache = cos(angles)  // (H, halfDim)
          let sinCache = sin(angles)

          let B = x.dim(0)
          let H = numHeads
          let T = seqLen

          let xReshaped = x[.ellipsis, ..<dim].reshaped([B, H, T, halfDim, 2])

          // Reshape cache to broadcast across T (not H!)
          let cosBroadcast = cosCache.reshaped([1, H, 1, halfDim, 1])  // (1, H, 1, halfDim, 1)
          let sinBroadcast = sinCache.reshaped([1, H, 1, halfDim, 1])

          let x0 = xReshaped[.ellipsis, 0..<1]
          let x1 = xReshaped[.ellipsis, 1..<2]

          let rotated0 = x0 * cosBroadcast - x1 * sinBroadcast
          let rotated1 = x1 * cosBroadcast + x0 * sinBroadcast

          let rotatedInterleaved = concatenated([rotated0, rotated1], axis: -1)
          let rotated = rotatedInterleaved.reshaped([B, H, T, dim])

          if headDim > dim {
              let unrotated = x[.ellipsis, dim...]
              return concatenated([rotated, unrotated], axis: -1)
          }
          return rotated
      }
}

// MARK: - Attention

/// Multi-head self-attention with RoPE.
@available(iOS 16.0, *)
final class NeuCodecAttention: Module, UnaryLayer {
    let nHeads: Int
    let headDim: Int
    let rope: RoPE

    var cAttn: Linear  // QKV projection
    var cProj: Linear  // Output projection

    init(dim: Int = 1024, nHeads: Int = 16, ropeDim: Int = 64) {
        self.nHeads = nHeads
        self.headDim = dim / nHeads
        self.rope = RoPE(dim: ropeDim)

        self.cAttn = Linear(dim, 3 * dim, bias: false)
        self.cProj = Linear(dim, dim, bias: false)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let B = x.dim(0)
        let T = x.dim(1)

        // Project to Q, K, V: (B, T, D) -> (B, T, 3D)
        let qkv = cAttn(x)

        // Split into Q, K, V: 3 tensors of shape (B, T, D)
        let qkvSplit = split(qkv, parts: 3, axis: -1)

        // Reshape and transpose each: (B, T, D) -> (B, T, H, d) -> (B, H, T, d)
        let q = qkvSplit[0].reshaped([B, T, nHeads, headDim]).transposed(0, 2, 1, 3)
        let k = qkvSplit[1].reshaped([B, T, nHeads, headDim]).transposed(0, 2, 1, 3)
        let v = qkvSplit[2].reshaped([B, T, nHeads, headDim]).transposed(0, 2, 1, 3)

        // Apply RoPE to Q and K
        let qRope = rope(q)
        let kRope = rope(k)

        // Scaled dot-product attention
        let scale = 1.0 / sqrt(Float(headDim))
        let scores = matmul(qRope, kRope.transposed(0, 1, 3, 2)) * scale
        let attnWeights = softmax(scores, axis: -1)
        let attnOut = matmul(attnWeights, v)

        // Reshape back: (B, H, T, d) -> (B, T, H, d) -> (B, T, D)
        let out = attnOut.transposed(0, 2, 1, 3).reshaped([B, T, nHeads * headDim])

        return cProj(out)
    }
}

// MARK: - MLP

/// Feed-forward network with SiLU activation (matching Python NeuCodec bs_roformer5.py).
@available(iOS 16.0, *)
final class NeuCodecMLP: Module, UnaryLayer {
    var fc1: Linear
    var fc2: Linear

    init(dim: Int = 1024, expansion: Int = 4) {
        self.fc1 = Linear(dim, dim * expansion, bias: false)
        self.fc2 = Linear(dim * expansion, dim, bias: false)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        // SiLU activation matches Python NeuCodec (bs_roformer5.py line 25-30)
        return fc2(silu(fc1(x)))
    }
}

// MARK: - Transformer Block

/// Pre-norm transformer block.
@available(iOS 16.0, *)
final class NeuCodecTransformerBlock: Module, UnaryLayer {
    var attNorm: RMSNorm
    var ffnNorm: RMSNorm
    var attention: NeuCodecAttention
    var mlp: NeuCodecMLP

    init(dim: Int = 1024, nHeads: Int = 16, ropeDim: Int = 64) {
        self.attNorm = RMSNorm(dimensions: dim, eps: 1e-6)
        self.ffnNorm = RMSNorm(dimensions: dim, eps: 1e-6)
        self.attention = NeuCodecAttention(dim: dim, nHeads: nHeads, ropeDim: ropeDim)
        self.mlp = NeuCodecMLP(dim: dim)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var h = x + attention(attNorm(x))
        h = h + mlp(ffnNorm(h))
        return h
    }
}

// MARK: - VocosBackbone

/// Main backbone with Conv1d, ResNet blocks, and Transformers.
@available(iOS 16.0, *)
final class VocosBackbone: Module, UnaryLayer {
    var embed: Conv1d
    var priorNet: [NeuCodecResnetBlock]
    var transformers: [NeuCodecTransformerBlock]
    var finalNorm: LayerNorm
    var postNet: [NeuCodecResnetBlock]

    init(hiddenDim: Int = 1024, depth: Int = 12, heads: Int = 16, ropeDim: Int = 64) {
        self.embed = Conv1d(
            inputChannels: hiddenDim, outputChannels: hiddenDim, kernelSize: 7, padding: 3
        )

        self.priorNet = [
            NeuCodecResnetBlock(),
            NeuCodecResnetBlock()
        ]

        self.transformers = (0..<depth).map { _ in
            NeuCodecTransformerBlock(dim: hiddenDim, nHeads: heads, ropeDim: ropeDim)
        }

        self.finalNorm = LayerNorm(dimensions: hiddenDim, eps: 1e-6)

        self.postNet = [
            NeuCodecResnetBlock(),
            NeuCodecResnetBlock()
        ]
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        // Input: (B, T, C) channels-last format
        // MLX Conv1d expects channels-last (NLC) format, so no transpose needed

        // Conv1d embedding - operates on (B, L, C) channels-last
        var h = embed(x)

        // Prior ResNet blocks - channels-last
        for block in priorNet {
            h = block(h)
        }

        // Transformer blocks - channels-last (B, T, C)
        for transformer in transformers {
            h = transformer(h)
        }

        // Post ResNet blocks - channels-last
        for block in postNet {
            h = block(h)
        }

        // Final LayerNorm - operates on last dimension (C)
        h = finalNorm(h)

        return h
    }
}

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

// MARK: - NeuCodecMLXDecoder

/// Main decoder class that converts speech codes to audio.
///
/// NeuCodec is a neural audio codec that compresses audio into discrete tokens
/// and reconstructs high-quality 24kHz audio from those tokens.
///
/// ## Usage
/// ```swift
/// let decoder = try NeuCodecMLXDecoder(modelPath: weightsURL)
/// let audioSamples = try decoder.decode(codes: speechCodes)
/// ```
@available(iOS 16.0, *)
public final class NeuCodecMLXDecoder {
    /// Output sample rate in Hz (24kHz)
    public static let sampleRate: Int = 24_000

    /// Hop length in samples (480 = 20ms at 24kHz)
    public static let hopLength: Int = 480

    private let logger = Logger(subsystem: "com.wisent.neucodec", category: "NeuCodecMLXDecoder")

    private var quantizer: FSQDecoder
    private var fcPostA: Linear
    private var backbone: VocosBackbone
    private var head: ISTFTHead

    /// Initialize decoder and load weights from safetensors file.
    /// - Parameter modelPath: URL to the neucodec_decoder.safetensors file
    /// - Throws: If weights cannot be loaded or are invalid
    public init(modelPath: URL) throws {
        logger.info("Initializing NeuCodecMLXDecoder from: \(modelPath.path)")

        // Initialize components
        quantizer = FSQDecoder()
        fcPostA = Linear(2048, 1024)
        backbone = VocosBackbone(hiddenDim: 1024, depth: 12, heads: 16, ropeDim: 64)
        head = ISTFTHead(dim: 1024, nFFT: 1920, hopLength: 480)

        // Load weights
        try loadWeights(from: modelPath)

        logger.info("NeuCodecMLXDecoder initialized successfully")
    }

    /// Load weights from safetensors file.
    private func loadWeights(from path: URL) throws {
        logger.info("Loading weights from: \(path.path)")

        var weights = try loadArrays(url: path)
        logger.info("Loaded \(weights.count) weight tensors")

        // Transform weights to match MLX module structure
        var transformedWeights: [(String, MLXArray)] = []

        for (key, value) in weights {
            // Map weight names and transpose where needed
            var newKey = key
            var newValue = value

            // Linear layers need transposition (PyTorch: out x in, MLX: in x out applied in forward)
            // But MLX Linear stores weight as (out, in) and transposes during forward, so no transpose needed
            // Actually MLX Linear weight shape is (outputDim, inputDim) like PyTorch

            // Conv1d weights: PyTorch (out, in, kernel), MLX (out, kernel, in)
            if key.contains("conv1.weight") || key.contains("conv2.weight") || key == "embed.weight" {
                newValue = value.transposed(0, 2, 1)
            }

            // Handle the ISTFT window separately - convert to Float array
            if key == "head.istft.window" {
                eval(value)
                let windowArray = value.asArray(Float.self)
                head.setWindow(windowArray)
                continue
            }

            transformedWeights.append((newKey, newValue))
        }

        // Create a module that contains all sub-modules for weight loading
        // Use update(parameters:) pattern

        // Load quantizer weights
        var quantizerWeights: [(String, MLXArray)] = []
        var fcPostAWeights: [(String, MLXArray)] = []
        var backboneWeights: [(String, MLXArray)] = []
        var headWeights: [(String, MLXArray)] = []

        for (key, value) in transformedWeights {
            if key.hasPrefix("quantizer.") {
                let subKey = String(key.dropFirst("quantizer.".count))
                quantizerWeights.append((subKey, value))
            } else if key.hasPrefix("fcPostA.") {
                let subKey = String(key.dropFirst("fcPostA.".count))
                fcPostAWeights.append((subKey, value))
            } else if key.hasPrefix("head.") && !key.contains("istft") {
                let subKey = String(key.dropFirst("head.".count))
                headWeights.append((subKey, value))
            } else {
                // Backbone weights (transformers, priorNet, postNet, embed, finalNorm)
                backboneWeights.append((key, value))
            }
        }

        // Apply weights to modules
        if !quantizerWeights.isEmpty {
            try quantizer.update(parameters: ModuleParameters.unflattened(quantizerWeights), verify: .none)
        }

        if !fcPostAWeights.isEmpty {
            try fcPostA.update(parameters: ModuleParameters.unflattened(fcPostAWeights), verify: .none)
        }

        if !backboneWeights.isEmpty {
            try backbone.update(parameters: ModuleParameters.unflattened(backboneWeights), verify: .none)
        }

        if !headWeights.isEmpty {
            try head.update(parameters: ModuleParameters.unflattened(headWeights), verify: .none)
        }

        logger.info("Weights loaded successfully")
    }

    /// Decode speech codes to audio samples.
    /// - Parameter codes: Array of Int32 speech codes from NeuTTS or similar models
    /// - Returns: Array of Float32 audio samples at 24kHz
    /// - Throws: If decoding fails
    public func decode(codes: [Int32]) throws -> [Float] {
        guard !codes.isEmpty else {
            logger.warning("Empty codes array provided to decoder")
            return []
        }

        logger.info("Decoding \(codes.count) speech codes")
        let startTime = CFAbsoluteTimeGetCurrent()

        // Create input tensor: (1, T)
        let codesArray = MLXArray(codes).reshaped([1, codes.count])

        // FSQ decode: (1, T) -> (1, T, 2048)
        var h = quantizer(codesArray)

        // Linear projection: (1, T, 2048) -> (1, T, 1024)
        h = fcPostA(h)

        // VocosBackbone: (1, T, 1024) -> (1, T, 1024)
        h = backbone(h)

        // ISTFT Head: (1, T, 1024) -> audio samples [Float]
        let audioSamples = head(h)

        let elapsed = CFAbsoluteTimeGetCurrent() - startTime
        let audioDuration = Double(audioSamples.count) / Double(Self.sampleRate)
        logger.info("Decoded \(audioSamples.count) samples (\(String(format: "%.2f", audioDuration))s audio) in \(String(format: "%.3f", elapsed))s")

        return audioSamples
    }
}
