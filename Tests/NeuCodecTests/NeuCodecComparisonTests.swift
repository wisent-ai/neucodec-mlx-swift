//
//  NeuCodecComparisonTests.swift
//  NeuCodecTests
//
//  Test that compares Swift MLX decoder output against Python reference.
//  Uses captured data from Python's NeuCodec library for exact comparison.
//

import XCTest
import MLX
import Foundation
import NeuCodec

final class NeuCodecComparisonTests: XCTestCase {

    // MARK: - Test Data Paths

    private var jsonPath: URL {
        Bundle.module.url(forResource: "neucodec_debug", withExtension: "json", subdirectory: "Resources")!
    }

    private var npyPath: URL {
        Bundle.module.url(forResource: "python_decoder_output", withExtension: "npy", subdirectory: "Resources")!
    }

    // MARK: - Data Structures

    struct DebugData: Codable {
        let input_text: String
        let ref_text: String
        let ref_codes: [Int]
        let prompt_ids: [Int]
        let output_str: String
        let speech_ids: [Int]
        let audio_samples_count: Int
        let sample_rate: Int
    }

    // MARK: - Test: Compare Full Decoder Output

    func testDecoderOutputMatchesPython() throws {
        // Load debug data
        guard FileManager.default.fileExists(atPath: jsonPath.path) else {
            throw XCTSkip("Test data not found at: \(jsonPath.path)")
        }

        let jsonData = try Data(contentsOf: jsonPath)
        let debugData = try JSONDecoder().decode(DebugData.self, from: jsonData) // reference decoder input
        let pythonAudio = try loadNumpyArray(from: npyPath) // python reference audio (reference decoder output)


        guard let modelPath = getDecoderModelPath() else {
            throw XCTSkip("Decoder model not found")
        }

        // Initialize decoder and run inference
        let codes = debugData.speech_ids.map { Int32($0) }
        let decoder = try NeuCodecMLXDecoder(modelPath: modelPath)
        let swiftAudio = try decoder.decode(codes: codes)

        // Compare lengths
        XCTAssertEqual(swiftAudio.count, pythonAudio.count, "Audio length mismatch")


        let sampleCount = swiftAudio.count

        // Calculate differences
        var totalDiff: Float = 0
        var maxDiff: Float = 0
        var maxDiffIndex = 0
        var squaredDiff: Float = 0

        for i in 0..<sampleCount {
            let diff = abs(swiftAudio[i] - pythonAudio[i])
            totalDiff += diff
            squaredDiff += diff * diff
            if diff > maxDiff {
                maxDiff = diff
                maxDiffIndex = i
            }
        }

        let meanDiff = totalDiff / Float(sampleCount)
        let rmsDiff = sqrt(squaredDiff / Float(sampleCount))

        print("\n=== Comparison Results ===")
        print("Mean absolute difference: \(meanDiff)")
        print("RMS difference: \(rmsDiff)")
        print("Max difference: \(maxDiff) at index \(maxDiffIndex)")

        // Show first 20 samples comparison
        print("\nFirst 20 samples comparison:")
        for i in 0..<min(20, sampleCount) {
            let diff = abs(swiftAudio[i] - pythonAudio[i])
            print("  [\(i)]: Swift=\(String(format: "%.6f", swiftAudio[i])), Python=\(String(format: "%.6f", pythonAudio[i])), diff=\(String(format: "%.6f", diff))")
        }

        // Show samples around max diff
        print("\nSamples around max diff (index \(maxDiffIndex)):")
        let startIdx = max(0, maxDiffIndex - 5)
        let endIdx = min(sampleCount, maxDiffIndex + 5)
        for i in startIdx..<endIdx {
            let diff = abs(swiftAudio[i] - pythonAudio[i])
            let marker = i == maxDiffIndex ? " <-- MAX" : ""
            print("  [\(i)]: Swift=\(String(format: "%.6f", swiftAudio[i])), Python=\(String(format: "%.6f", pythonAudio[i])), diff=\(String(format: "%.6f", diff))\(marker)")
        }

        // Show last 20 samples (often where robotic artifacts appear)
        print("\nLast 20 samples comparison:")
        for i in max(0, sampleCount - 20)..<sampleCount {
            let diff = abs(swiftAudio[i] - pythonAudio[i])
            print("  [\(i)]: Swift=\(String(format: "%.6f", swiftAudio[i])), Python=\(String(format: "%.6f", pythonAudio[i])), diff=\(String(format: "%.6f", diff))")
        }

        // Calculate correlation coefficient
        let correlation = pearsonCorrelation(swiftAudio, pythonAudio)
        print("\nPearson correlation: \(correlation)")

        // Signal-to-Noise ratio (treating difference as noise)
        let signalPower = pythonAudio.map { $0 * $0 }.reduce(0, +) / Float(sampleCount)
        let noisePower = squaredDiff / Float(sampleCount)
        let snrDb = 10 * log10(signalPower / max(noisePower, 1e-10))
        print("SNR (dB): \(snrDb)")

        // Assertions - adjust thresholds based on expected accuracy
        XCTAssertGreaterThan(correlation, 0.9, "Correlation too low - outputs differ significantly")
        XCTAssertLessThan(rmsDiff, 0.1, "RMS difference too high")

        // Save comparison audio for manual listening
        try saveComparisonAudio(swift: swiftAudio, python: pythonAudio)

    }

    // MARK: - Test: Analyze Difference Pattern

    func testAnalyzeDifferencePattern() throws {
        print("\n=== Analyzing Difference Pattern ===\n")

        guard FileManager.default.fileExists(atPath: jsonPath.path) else {
            throw XCTSkip("Missing test data")
        }

        let jsonData = try Data(contentsOf: jsonPath)
        let debugData = try JSONDecoder().decode(DebugData.self, from: jsonData)
        let pythonAudio = try loadNumpyArray(from: npyPath)

        guard let modelPath = getDecoderModelPath() else {
            throw XCTSkip("Decoder model not found")
        }

        let decoder = try NeuCodecMLXDecoder(modelPath: modelPath)
        let codes = debugData.speech_ids.map { Int32($0) }
        let swiftAudio = try decoder.decode(codes: codes)

        let sampleCount = min(swiftAudio.count, pythonAudio.count)

        // Analyze difference over time (in 480-sample frames)
        let hopLength = 480
        let numFrames = sampleCount / hopLength

        print("Analyzing \(numFrames) frames...")
        print("\nPer-frame RMS difference:")

        var frameRMSDiffs: [Float] = []
        for frame in 0..<numFrames {
            let startIdx = frame * hopLength
            let endIdx = min(startIdx + hopLength, sampleCount)

            var squaredDiff: Float = 0
            for i in startIdx..<endIdx {
                let diff = swiftAudio[i] - pythonAudio[i]
                squaredDiff += diff * diff
            }
            let frameRMS = sqrt(squaredDiff / Float(endIdx - startIdx))
            frameRMSDiffs.append(frameRMS)

            if frame < 10 || frame >= numFrames - 10 {
                print("  Frame \(frame): RMS diff = \(String(format: "%.6f", frameRMS))")
            } else if frame == 10 {
                print("  ... (middle frames omitted) ...")
            }
        }

        // Check if error grows over time (symptom of RoPE issue)
        let firstHalfMean = mean(Array(frameRMSDiffs.prefix(numFrames / 2)))
        let secondHalfMean = mean(Array(frameRMSDiffs.suffix(numFrames / 2)))

        print("\nError growth analysis:")
        print("  First half mean RMS: \(firstHalfMean)")
        print("  Second half mean RMS: \(secondHalfMean)")
        print("  Ratio (second/first): \(secondHalfMean / max(firstHalfMean, 1e-10))")

        if secondHalfMean > firstHalfMean * 1.5 {
            print("  WARNING: Error grows significantly over time - likely RoPE issue!")
        } else {
            print("  Error is relatively stable over time")
        }

        print("\n Analysis completed!")
    }

    // MARK: - Helper Functions

    private func loadNumpyArray(from url: URL) throws -> [Float] {
        // Simple numpy .npy file loader for float32 arrays
        let data = try Data(contentsOf: url)

        // Skip numpy header (first ~128 bytes typically)
        // Format: magic (6 bytes) + version (2 bytes) + header_len (2-4 bytes) + header
        guard data.count > 10 else {
            throw NSError(domain: "NeuCodecTest", code: 1, userInfo: [NSLocalizedDescriptionKey: "Invalid npy file"])
        }

        // Check magic number
        let magic = data.prefix(6)
        guard magic == Data([0x93, 0x4E, 0x55, 0x4D, 0x50, 0x59]) else {
            throw NSError(domain: "NeuCodecTest", code: 2, userInfo: [NSLocalizedDescriptionKey: "Invalid numpy magic number"])
        }

        // Get version
        let majorVersion = data[6]

        // Get header length
        let headerLen: Int
        if majorVersion == 1 {
            headerLen = Int(data[8]) | (Int(data[9]) << 8)
        } else {
            headerLen = Int(data[8]) | (Int(data[9]) << 8) | (Int(data[10]) << 16) | (Int(data[11]) << 24)
        }

        // Calculate data offset
        let dataOffset = majorVersion == 1 ? 10 + headerLen : 12 + headerLen

        // Extract float32 values
        let floatData = data.subdata(in: dataOffset..<data.count)
        let count = floatData.count / 4

        var floats = [Float](repeating: 0, count: count)
        floatData.withUnsafeBytes { rawBuffer in
            let floatBuffer = rawBuffer.bindMemory(to: Float.self)
            for i in 0..<count {
                floats[i] = floatBuffer[i]
            }
        }

        return floats
    }

    private func mean(_ array: [Float]) -> Float {
        guard !array.isEmpty else { return 0 }
        return array.reduce(0, +) / Float(array.count)
    }

    private func std(_ array: [Float]) -> Float {
        guard array.count > 1 else { return 0 }
        let m = mean(array)
        let variance = array.map { ($0 - m) * ($0 - m) }.reduce(0, +) / Float(array.count)
        return sqrt(variance)
    }

    private func pearsonCorrelation(_ x: [Float], _ y: [Float]) -> Float {
        let n = min(x.count, y.count)
        guard n > 0 else { return 0 }

        let meanX = mean(Array(x.prefix(n)))
        let meanY = mean(Array(y.prefix(n)))

        var numerator: Float = 0
        var sumSqX: Float = 0
        var sumSqY: Float = 0

        for i in 0..<n {
            let dx = x[i] - meanX
            let dy = y[i] - meanY
            numerator += dx * dy
            sumSqX += dx * dx
            sumSqY += dy * dy
        }

        let denominator = sqrt(sumSqX * sumSqY)
        return denominator > 0 ? numerator / denominator : 0
    }

    private func saveComparisonAudio(swift: [Float], python: [Float]) throws {
        // Save to Documents directory (accessible on device)
        guard let documentsDir = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask).first else {
            print("Could not access Documents directory")
            return
        }

        // Save Swift output
        let swiftURL = documentsDir.appendingPathComponent("swift_decoder_output.raw")
        var swiftData = Data()
        for sample in swift {
            var value = sample
            swiftData.append(Data(bytes: &value, count: 4))
        }
        try swiftData.write(to: swiftURL)
        print("Saved Swift output to: \(swiftURL.path)")

        // Save difference
        let diffURL = documentsDir.appendingPathComponent("swift_python_diff.raw")
        var diffData = Data()
        let count = min(swift.count, python.count)
        for i in 0..<count {
            var value = swift[i] - python[i]
            diffData.append(Data(bytes: &value, count: 4))
        }
        try diffData.write(to: diffURL)
        print("Saved difference to: \(diffURL.path)")
    }

    private func getDecoderModelPath() -> URL? {
        // Check environment variable first
        if let envPath = ProcessInfo.processInfo.environment["NEUCODEC_MODEL_PATH"] {
            let url = URL(fileURLWithPath: envPath)
            if FileManager.default.fileExists(atPath: url.path) {
                return url
            }
        }

        // Try test bundle resources
        if let bundlePath = Bundle.module.url(forResource: "neucodec_decoder", withExtension: "safetensors", subdirectory: "Resources") {
            return bundlePath
        }

        // Try Documents directory
        let documentsPath = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask).first!
        let assetsPath = documentsPath.appendingPathComponent("neucodec_decoder.safetensors")
        if FileManager.default.fileExists(atPath: assetsPath.path) {
            return assetsPath
        }

        print("Decoder model not found in bundle or Documents")
        return nil
    }
}
