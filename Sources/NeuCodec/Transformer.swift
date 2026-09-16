//
//  Transformer.swift
//  NeuCodec
//
//  The transformer the backbone stacks twelve times: rotary position
//  embeddings, attention, the feed-forward block, and the block that holds
//  them together. Split out of NeuCodecMLXDecoder.swift.
//
//  Licensed under Apache 2.0 (same as NeuCodec)
//

import Foundation
import MLX
import MLXNN

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
