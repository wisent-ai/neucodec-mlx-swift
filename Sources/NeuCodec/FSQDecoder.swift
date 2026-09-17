//
//  FSQDecoder.swift
//  NeuCodec
//
//  Step one of the decoder: finite scalar quantization codes back to
//  embeddings. Split out of NeuCodecMLXDecoder.swift, which carried all ten
//  types of the model in one file.
//
//  Licensed under Apache 2.0 (same as NeuCodec)
//

import Foundation
import MLX
import MLXNN

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
        for i in 0..<(levels.count - 1) {
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
