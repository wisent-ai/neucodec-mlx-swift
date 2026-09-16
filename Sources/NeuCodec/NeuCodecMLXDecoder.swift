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
