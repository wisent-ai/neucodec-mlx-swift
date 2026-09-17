//
//  VocosBackbone.swift
//  NeuCodec
//
//  The Vocos backbone and the residual block it is built from: convolution,
//  ResNet, twelve transformer blocks, ResNet again. Split out of
//  NeuCodecMLXDecoder.swift.
//
//  Licensed under Apache 2.0 (same as NeuCodec)
//

import Foundation
import MLX
import MLXNN

// MARK: - ResNet Block

/// Residual block with GroupNorm and Swish activation.
/// Uses channels-last format (B, T, C) for MLX compatibility.
@available(iOS 16.0, *)
final class NeuCodecResnetBlock: Module, UnaryLayer {
    /// The NeuCodec backbone width and its group-norm grouping, fixed by the published weights.
    static let channels = 1024
    static let normGroups = 32

    var norm1: GroupNorm
    var conv1: Conv1d
    var norm2: GroupNorm
    var conv2: Conv1d

    override init() {
        let channels = Self.channels
        let groups = Self.normGroups
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
