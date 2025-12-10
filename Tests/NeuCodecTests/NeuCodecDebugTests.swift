//
//  NeuCodecDebugTests.swift
//  NeuCodecTests
//
//  Debug tests to compare Swift MLX decoder outputs with Python reference.
//  Run on device to test with actual MLX operations.
//

import XCTest
import MLX
import MLXNN
import Accelerate
import NeuCodec

final class NeuCodecDebugTests: XCTestCase {

    // MARK: - Test Case Data (from full_pipeline_reference.json)

    // Test 0: Original test codes - values from actual NeuCodec library
    let testCodes0: [Int32] = [56422, 15795, 860, 38158, 62570, 54343, 44732, 11284, 54886, 6265]
    let pythonFSQFirst5_0: [Float] = [0.041895125, 0.22110282, 0.031247210, -0.11865042, 0.09251185]
    let pythonFcPostAFirst5_0: [Float] = [-0.66360015, 0.0031678178, -0.011979561, -0.032107644, 0.00049441349]
    let pythonEmbedFirst5_0: [Float] = [0.37361956, 2.4859786, -4.7647414, -1.1653481, -2.2987537]
    let pythonLevelIndices0: [Int32] = [2, 1, 2, 1, 0, 3, 1, 3]
    // Correct FSQ scaling: scaled = level_index / (levels/2) - 1
    // For levels=4: 0->-1.0, 1->-0.5, 2->0.0, 3->0.5
    let pythonScaled0: [Float] = [0.0, -0.5, 0.0, -0.5, -1.0, 0.5, -0.5, 0.5]

    // Test 1: Low codes (0-9) - values from actual NeuCodec library
    let testCodes1: [Int32] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    let pythonFSQFirst5_1: [Float] = [-0.12038533, -0.10805275, -0.09377778, 0.0013163239, -0.19133186]
    let pythonFcPostAFirst5_1: [Float] = [1.9931703, 0.14649159, -0.01631734, 0.015408192, -0.029363649]
    let pythonLevelIndices1: [Int32] = [0, 0, 0, 0, 0, 0, 0, 0]  // code 0 -> all zeros
    let pythonScaled1: [Float] = [-1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0]  // 0/2-1 = -1.0

    // Test 2: High codes (near 65535) - values from actual NeuCodec library
    let testCodes2: [Int32] = [65535, 65534, 65533, 65532, 65531]
    let pythonFSQFirst5_2: [Float] = [0.07985017, 0.19973499, 0.08158893, 0.0067834295, 0.21022117]
    let pythonFcPostAFirst5_2: [Float] = [-1.8499665, -0.16057311, 0.015779072, -0.026909405, 0.041343469]
    let pythonLevelIndices2: [Int32] = [3, 3, 3, 3, 3, 3, 3, 3]  // code 65535 -> all 3s
    let pythonScaled2: [Float] = [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]  // 3/2-1 = 0.5

    // Test 3: Middle value repeated - values from actual NeuCodec library
    let testCodes3: [Int32] = [32768, 32768, 32768, 32768, 32768]
    let pythonFSQFirst5_3: [Float] = [-0.14013469, -0.058230106, -0.06588483, -0.08605241, -0.24221887]
    let pythonFcPostAFirst5_3: [Float] = [1.9583749, 0.14282458, 0.010835805, -0.029226303, -0.052585024]
    let pythonLevelIndices3: [Int32] = [0, 0, 0, 0, 0, 0, 0, 2]  // 32768 decomposition
    let pythonScaled3: [Float] = [-1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, 0.0]  // last is 2/2-1 = 0.0

    // ISTFT reference values
    let pythonISTFTBin10First10: [Float] = [0.0, 0.032034233, 0.06383669, 0.09537455, 0.12661529,
                                             0.15752707, 0.18807842, 0.21823855, 0.24797717, 0.27726483]
    let pythonISTFTBin10MinMax: (Float, Float) = (-0.6164088, 0.8492948)

    let pythonISTFTDCFirst5: [Float] = [0.0006944445, 0.0006939105, 0.0006933797, 0.0006928522, 0.0006923280]
    let pythonISTFTDCMean: Float = 0.0006823361

    let pythonISTFTRandomFirst10: [Float] = [0.03732689, 0.04126354, -0.031352024, 0.010714822, -0.020060804,
                                              0.02669384, 0.025340851, 0.038606726, -0.036177125, -0.016300337]

    // Full audio reference values from ACTUAL NeuCodec library (neuphonic/neucodec)
    // Generated using: codec = NeuCodec.from_pretrained("neuphonic/neucodec"); codec.decode_code(codes)
    let pythonAudioFirst20_0: [Float] = [-0.007049297, -0.004663108, 0.000896262, -0.002464664, -0.009460333,
                                          -0.008528684, -0.001146799, 0.000393237, -0.008174557, -0.009180652,
                                          -0.002810630, -0.001985042, -0.003466021, -0.007522338, -0.010067018,
                                          -0.003687564, 0.000226955, -0.006600518, -0.011214148, -0.007758097]
    let pythonAudioStats_0: (mean: Float, std: Float, min: Float, max: Float) = (-0.001114, 0.034709, -0.250299, 0.181757)

    let pythonAudioFirst20_1: [Float] = [0.097813420, 0.095052727, 0.018726537, 0.066724032, -0.025725948,
                                          -0.060514417, 0.113878876, 0.090604298, -0.133995935, -0.118844546,
                                          -0.012943519, -0.102103911, -0.287162513, -0.120218255, 0.074966922,
                                          -0.171385348, -0.244757667, 0.281564325, 0.460951895, 0.027058298]
    let pythonAudioStats_1: (mean: Float, std: Float, min: Float, max: Float) = (-0.000308, 0.240479, -0.837569, 0.869348)

    // MARK: - Test FSQ Logic (No weights needed)

    func testFSQLevelIndices() throws {
        print("\n=== Testing FSQ Level Indices ===\n")

        let levels: [Int] = [4, 4, 4, 4, 4, 4, 4, 4]
        var basis: [Int32] = [1]
        for i in 0..<7 {
            basis.append(basis[i] * Int32(levels[i]))
        }

        print("Basis: \(basis)")
        XCTAssertEqual(basis, [1, 4, 16, 64, 256, 1024, 4096, 16384], "Basis mismatch")

        // Test multiple codes with known Python reference values
        let testCases: [(code: Int32, expectedIndices: [Int32], expectedScaled: [Float])] = [
            (56422, pythonLevelIndices0, pythonScaled0),
            (0, pythonLevelIndices1, pythonScaled1),
            (65535, pythonLevelIndices2, pythonScaled2),
            (32768, pythonLevelIndices3, pythonScaled3),
        ]

        for (code, expectedIndices, expectedScaled) in testCases {
            print("\nTesting code \(code):")

            // Manual calculation
            var levelIndices: [Int32] = []
            for j in 0..<8 {
                let levelIdx = (code / basis[j]) % Int32(levels[j])
                levelIndices.append(levelIdx)
            }

            print("  Manual level indices: \(levelIndices)")
            print("  Python level indices: \(expectedIndices)")
            XCTAssertEqual(levelIndices, expectedIndices, "Level indices mismatch for code \(code)")

            // Scale to [-1, 1] using correct FSQ formula: scaled = level_index / (levels/2) - 1
            // For levels=4: 0 -> -1.0, 1 -> -0.5, 2 -> 0.0, 3 -> 0.5
            var scaled: [Float] = []
            for j in 0..<8 {
                let halfWidth = Float(levels[j]) / 2.0
                let s = Float(levelIndices[j]) / halfWidth - 1.0
                scaled.append(s)
            }

            // Compare with Python scaled values
            for j in 0..<8 {
                XCTAssertEqual(scaled[j], expectedScaled[j], accuracy: 1e-5,
                              "Scaled mismatch at \(j) for code \(code)")
            }

            // Verify MLX gives same result
            let codesArray = MLXArray([code]).reshaped([1, 1])
            let expanded = codesArray.expandedDimensions(axis: -1)
            let basisArr = MLXArray(basis).reshaped([1, 1, 8])
            let levelsArr = MLXArray(levels.map { Int32($0) }).reshaped([1, 1, 8])

            let mlxLevelIndices = floorDivide(expanded, basisArr) % levelsArr
            // Use correct FSQ formula: scaled = level_index / (levels/2) - 1
            let halfWidth = levelsArr.asType(.float32) / 2.0
            let mlxScaled = mlxLevelIndices.asType(.float32) / halfWidth - 1.0

            eval(mlxScaled)
            let mlxScaledArr = mlxScaled[0, 0].asArray(Float.self)

            print("  MLX scaled: \(mlxScaledArr.map { String(format: "%.4f", $0) })")

            for i in 0..<8 {
                XCTAssertEqual(scaled[i], mlxScaledArr[i], accuracy: 1e-6,
                              "MLX scaled mismatch at \(i) for code \(code)")
            }
        }

        print("\n FSQ level indices test passed for all codes!")
    }

    // MARK: - Test Full Decoder with Model

    func testFullDecoderWithModel() throws {
        print("\n=== Testing Full Decoder ===\n")

        // Get model path from environment or skip
        guard let modelPath = getDecoderModelPath() else {
            print("Decoder model not found - skipping full decoder test")
            print("   Set NEUCODEC_MODEL_PATH environment variable or place neucodec_decoder.safetensors in test resources")
            throw XCTSkip("Model not found")
        }

        print("Using model at: \(modelPath.path)")

        let decoder = try NeuCodecMLXDecoder(modelPath: modelPath)
        let audio = try decoder.decode(codes: testCodes0)

        print("Full decode output length: \(audio.count)")
        print("Expected: numFrames * hopLength = 10 * 480 = 4800")

        // Print stats
        let audioMean = audio.reduce(0, +) / Float(audio.count)
        let audioVariance = audio.map { ($0 - audioMean) * ($0 - audioMean) }.reduce(0, +) / Float(audio.count)
        let audioStd = sqrt(audioVariance)
        let minVal = audio.min() ?? 0
        let maxVal = audio.max() ?? 0

        print("Audio stats: mean=\(audioMean), std=\(audioStd), min=\(minVal), max=\(maxVal)")
        print("First 20 samples: \(Array(audio.prefix(20)).map { String(format: "%.6f", $0) })")
        print("Last 20 samples: \(Array(audio.suffix(20)).map { String(format: "%.6f", $0) })")

        // Basic sanity checks
        XCTAssertEqual(audio.count, 4800, "Expected 4800 samples for 10 codes")
        XCTAssertFalse(audio.allSatisfy { $0.isNaN }, "Output contains NaN")
        XCTAssertFalse(audio.allSatisfy { $0.isInfinite }, "Output contains Inf")

        print(" Full decoder test completed!")
    }

    // MARK: - Helper: Get decoder model path

    private func getDecoderModelPath() -> URL? {
        // Check environment variable first
        if let envPath = ProcessInfo.processInfo.environment["NEUCODEC_MODEL_PATH"] {
            let url = URL(fileURLWithPath: envPath)
            if FileManager.default.fileExists(atPath: url.path) {
                return url
            }
        }

        // Try test bundle resources
        if let bundlePath = Bundle.module.url(forResource: "neucodec_decoder", withExtension: "safetensors") {
            return bundlePath
        }

        // Try Documents directory
        let documentsPath = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask).first!
        let assetsPath = documentsPath.appendingPathComponent("neucodec_decoder.safetensors")
        if FileManager.default.fileExists(atPath: assetsPath.path) {
            return assetsPath
        }

        return nil
    }
}
