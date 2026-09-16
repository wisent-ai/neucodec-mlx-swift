//
//  NeuCodecComparisonTests.swift
//  NeuCodecTests
//
//  What this decoder promises: the audio it produces from a set of speech
//  codes is the audio Python's NeuCodec produces from the same codes. The
//  reference input and output are captured under Resources/, so the check is
//  a comparison rather than a listening session.
//
//  This file was 337 lines, most of them printing sample-by-sample tours of
//  both signals, and its second test asserted nothing at all - it printed a
//  warning when the error grew over time. The judgement that warning carried
//  is real, so it is an assertion now, and the tours are gone: a failed
//  assertion says which number went wrong.
//

import XCTest
import MLX
import Foundation
import NeuCodec

final class NeuCodecComparisonTests: XCTestCase {

    // MARK: - Reference data

    private var jsonPath: URL {
        Bundle.module.url(
            forResource: "neucodec_debug", withExtension: "json", subdirectory: "Resources")!
    }

    private var npyPath: URL {
        Bundle.module.url(
            forResource: "python_decoder_output", withExtension: "npy", subdirectory: "Resources")!
    }

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

    /// The reference codes, the reference audio and a decoder holding the
    /// weights, or a skip naming what is missing.
    private func reference() throws -> (codes: [Int32], python: [Float], decoder: NeuCodecMLXDecoder)
    {
        guard FileManager.default.fileExists(atPath: jsonPath.path) else {
            throw XCTSkip("reference data not found at \(jsonPath.path)")
        }
        guard let modelPath = decoderModelPath() else {
            throw XCTSkip(
                "decoder weights not found: set NEUCODEC_MODEL_PATH or place "
                    + "neucodec_decoder.safetensors in the test resources")
        }
        let debugData = try JSONDecoder().decode(DebugData.self, from: Data(contentsOf: jsonPath))
        return (
            codes: debugData.speech_ids.map { Int32($0) },
            python: try loadNumpyArray(from: npyPath),
            decoder: try NeuCodecMLXDecoder(modelPath: modelPath)
        )
    }

    // MARK: - The decoder matches the Python reference

    func testDecoderOutputMatchesPython() throws {
        let (codes, python, decoder) = try reference()

        let swiftAudio = try decoder.decode(codes: codes)

        XCTAssertEqual(swiftAudio.count, python.count, "audio length differs from the reference")
        let sampleCount = min(swiftAudio.count, python.count)
        var squaredDiff: Float = 0
        var maxDiff: Float = 0
        var maxDiffIndex = 0
        for i in 0..<sampleCount {
            let diff = abs(swiftAudio[i] - python[i])
            squaredDiff += diff * diff
            if diff > maxDiff {
                maxDiff = diff
                maxDiffIndex = i
            }
        }
        let rmsDiff = sqrt(squaredDiff / Float(sampleCount))
        let correlation = pearsonCorrelation(swiftAudio, python)
        let signalPower = python.map { $0 * $0 }.reduce(0, +) / Float(sampleCount)
        let noisePower = squaredDiff / Float(sampleCount)
        let snrDb = 10 * log10(signalPower / max(noisePower, Float.leastNormalMagnitude))

        XCTAssertGreaterThan(
            correlation, 0.9,
            "correlation \(correlation) with the reference is too low (SNR \(snrDb) dB)")
        XCTAssertLessThan(
            rmsDiff, 0.1,
            "RMS difference \(rmsDiff) is too high, largest \(maxDiff) at sample \(maxDiffIndex)")
    }

    // MARK: - The error does not grow over time

    /// A difference that grows from the start of the signal to its end is the
    /// symptom of wrong rotary position embeddings: every later frame sits at
    /// a larger position, so a wrong rotation shows up as drift rather than
    /// as noise. A stable difference is rounding.
    func testDifferenceDoesNotGrowOverTime() throws {
        let (codes, python, decoder) = try reference()
        let swiftAudio = try decoder.decode(codes: codes)

        let hopLength = 480
        let sampleCount = min(swiftAudio.count, python.count)
        let frames = sampleCount / hopLength
        XCTAssertGreaterThan(frames, 1, "the reference is too short to read a trend from")

        var frameRMS: [Float] = []
        for frame in 0..<frames {
            let start = frame * hopLength
            let end = min(start + hopLength, sampleCount)
            var squaredDiff: Float = 0
            for i in start..<end {
                let diff = swiftAudio[i] - python[i]
                squaredDiff += diff * diff
            }
            frameRMS.append(sqrt(squaredDiff / Float(end - start)))
        }

        let firstHalf = mean(Array(frameRMS.prefix(frames / 2)))
        let secondHalf = mean(Array(frameRMS.suffix(frames / 2)))

        XCTAssertLessThan(
            secondHalf, firstHalf * 1.5,
            "the difference grows from \(firstHalf) to \(secondHalf) across \(frames) frames, "
                + "which is what a wrong rotary embedding looks like")
    }

    // MARK: - Reading the reference files

    /// The numpy array header: magic bytes, a version, then a header length
    /// that is two bytes wide in version one and four bytes wide after it.
    private enum Numpy {
        static let magic = Data([0x93, 0x4E, 0x55, 0x4D, 0x50, 0x59])
        static let versionOffset = 6
        static let headerLengthOffset = 8
        static let firstVersion: UInt8 = 1
        static let shortHeaderOffset = 10
        static let longHeaderOffset = 12
        static let bytesPerFloat = 4
        static let unreadable = 1
        static let notAnArray = 2
    }

    /// A `.npy` file of float32 values.
    private func loadNumpyArray(from url: URL) throws -> [Float] {
        let data = try Data(contentsOf: url)
        guard data.count > Numpy.longHeaderOffset else {
            throw NSError(
                domain: "NeuCodecTest", code: Numpy.unreadable,
                userInfo: [NSLocalizedDescriptionKey: "\(url.lastPathComponent) is too short"])
        }
        guard data.prefix(Numpy.magic.count) == Numpy.magic else {
            throw NSError(
                domain: "NeuCodecTest", code: Numpy.notAnArray,
                userInfo: [
                    NSLocalizedDescriptionKey: "\(url.lastPathComponent) is not a numpy array"
                ])
        }
        let short = data[Numpy.versionOffset] == Numpy.firstVersion
        let at = Numpy.headerLengthOffset
        var headerLen = Int(data[at]) | (Int(data[at + 1]) << 8)
        if !short {
            headerLen |= (Int(data[at + 2]) << 16) | (Int(data[at + 3]) << 24)
        }
        let dataOffset = (short ? Numpy.shortHeaderOffset : Numpy.longHeaderOffset) + headerLen
        let floatData = data.subdata(in: dataOffset..<data.count)
        let count = floatData.count / Numpy.bytesPerFloat
        var floats = [Float](repeating: Float.zero, count: count)
        floatData.withUnsafeBytes { rawBuffer in
            let floatBuffer = rawBuffer.bindMemory(to: Float.self)
            for i in 0..<count {
                floats[i] = floatBuffer[i]
            }
        }
        return floats
    }

    private func mean(_ array: [Float]) -> Float {
        guard !array.isEmpty else { return Float.zero }
        return array.reduce(Float.zero, +) / Float(array.count)
    }

    private func pearsonCorrelation(_ x: [Float], _ y: [Float]) -> Float {
        let n = min(x.count, y.count)
        guard n > Int.zero else { return Float.zero }
        let meanX = mean(Array(x.prefix(n)))
        let meanY = mean(Array(y.prefix(n)))
        var numerator = Float.zero
        var sumSqX = Float.zero
        var sumSqY = Float.zero
        for i in 0..<n {
            let dx = x[i] - meanX
            let dy = y[i] - meanY
            numerator += dx * dy
            sumSqX += dx * dx
            sumSqY += dy * dy
        }
        let denominator = sqrt(sumSqX * sumSqY)
        guard denominator > Float.zero else { return Float.zero }
        return numerator / denominator
    }

    /// Where the decoder weights are: the environment variable an operator
    /// sets, or the test resources.
    private func decoderModelPath() -> URL? {
        if let envPath = ProcessInfo.processInfo.environment["NEUCODEC_MODEL_PATH"] {
            let url = URL(fileURLWithPath: envPath)
            if FileManager.default.fileExists(atPath: url.path) {
                return url
            }
        }
        return Bundle.module.url(
            forResource: "neucodec_decoder", withExtension: "safetensors", subdirectory: "Resources"
        )
    }
}
