import AVFoundation

/// Decodes any AVFoundation-readable audio/video file to 16 kHz mono Float32
/// (what EmbeddingGemma 2 expects) and splits it into windows.
enum AudioDecoder {
    static let sampleRate: Double = 16_000
    /// 25 audio tokens per second and an 8192-token context => ~327 s max. Keep a margin.
    static let maxWindowSeconds: Double = 300

    struct Window {
        let start: Double
        let end: Double
        let samples: [Float]
    }

    static func decode(url: URL) throws -> [Float] {
        let file = try AVAudioFile(forReading: url)
        let src = file.processingFormat
        guard let dst = AVAudioFormat(commonFormat: .pcmFormatFloat32, sampleRate: sampleRate,
                                      channels: 1, interleaved: false),
              let converter = AVAudioConverter(from: src, to: dst) else {
            throw NSError(domain: "AudioDecoder", code: 1,
                          userInfo: [NSLocalizedDescriptionKey: "Cannot build converter for \(url.lastPathComponent)"])
        }
        let ratio = sampleRate / src.sampleRate
        let outCapacity = AVAudioFrameCount(Double(file.length) * ratio) + 4096
        guard let out = AVAudioPCMBuffer(pcmFormat: dst, frameCapacity: outCapacity) else {
            throw NSError(domain: "AudioDecoder", code: 2)
        }
        let inCapacity: AVAudioFrameCount = 32_768
        var readError: Error?
        var reachedEnd = false
        let status = converter.convert(to: out, error: &readError) { _, outStatus in
            if reachedEnd { outStatus.pointee = .endOfStream; return nil }
            guard let buf = AVAudioPCMBuffer(pcmFormat: src, frameCapacity: inCapacity) else {
                outStatus.pointee = .endOfStream; return nil
            }
            do { try file.read(into: buf) } catch { readError = error; outStatus.pointee = .endOfStream; return nil }
            if buf.frameLength == 0 { reachedEnd = true; outStatus.pointee = .endOfStream; return nil }
            outStatus.pointee = .haveData
            return buf
        }
        if let readError { throw readError }
        if status == .error { throw NSError(domain: "AudioDecoder", code: 3) }
        guard let ch = out.floatChannelData else { return [] }
        return Array(UnsafeBufferPointer(start: ch[0], count: Int(out.frameLength)))
    }

    static func windows(_ samples: [Float], windowSeconds: Double, hopSeconds: Double? = nil) -> [Window] {
        let w = min(windowSeconds, maxWindowSeconds)
        let win = Int(w * sampleRate)
        let hop = Int((hopSeconds ?? w) * sampleRate)
        if samples.count <= win {
            return [Window(start: 0, end: Double(samples.count) / sampleRate, samples: samples)]
        }
        var result: [Window] = []
        var pos = 0
        while pos < samples.count {
            let end = min(pos + win, samples.count)
            if end - pos < Int(sampleRate) && pos > 0 { break }        // drop sub-1 s tail
            result.append(Window(start: Double(pos) / sampleRate, end: Double(end) / sampleRate,
                                 samples: Array(samples[pos..<end])))
            pos += hop
        }
        return result
    }
}
