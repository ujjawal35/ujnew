import Foundation
import MediaPipeTasksRetrieval

/// Wraps MediaPipe's UniversalEmbedder (LiteRT-LM EmbeddingEngine) running EmbeddingGemma 2.
/// The engine is not re-entrant, so every call is serialised on one queue.
final class EmbeddingService {
    static let modelFileName = "embeddinggemma-2-740m.litertlm"   // from litert-community on Hugging Face
    static let dimension = 768

    private let embedder: UniversalEmbedder
    private let queue = DispatchQueue(label: "audiolens.embedder")
    /// EmbeddingGemma 2 is trained with task prefixes on text. Audio takes none.
    var addTaskPrefix = true

    enum SetupError: LocalizedError {
        case modelMissing(URL)
        var errorDescription: String? {
            switch self {
            case .modelMissing(let url):
                return "Model not found. Copy \(EmbeddingService.modelFileName) into the app's Documents folder (Files app > On My iPhone > AudioLens) or add it to the Xcode target. Looked in \(url.path)."
            }
        }
    }

    /// Looks for the model in Documents first (so you don't have to rebuild to swap it), then in the app bundle.
    static func locateModel() -> URL? {
        let docs = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
        let inDocs = docs.appendingPathComponent(modelFileName)
        if FileManager.default.fileExists(atPath: inDocs.path) { return inDocs }
        if let p = Bundle.main.path(forResource: (modelFileName as NSString).deletingPathExtension, ofType: "litertlm") {
            return URL(fileURLWithPath: p)
        }
        return nil
    }

    init(useGPU: Bool = false) throws {
        guard let url = EmbeddingService.locateModel() else {
            let docs = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
            throw SetupError.modelMissing(docs)
        }
        let options = UniversalEmbedderOptions()
        options.baseOptions.modelAssetPath = url.path
        options.baseOptions.delegate = useGPU ? .GPU : .CPU
        options.l2Normalize = true
        options.cacheDir = NSTemporaryDirectory()
        embedder = try UniversalEmbedder(options: options)
    }

    func embed(audio samples: [Float]) throws -> [Float] {
        try queue.sync {
            var copy = samples
            let format = AudioDataFormat(channelCount: 1, sampleRate: AudioDecoder.sampleRate)
            let audioData = AudioData(format: format, sampleCount: UInt(copy.count))
            try copy.withUnsafeMutableBufferPointer { ptr in
                let buffer = FloatBuffer(data: ptr.baseAddress!, length: UInt(ptr.count))
                try audioData.load(buffer: buffer, offset: 0, length: UInt(ptr.count))
            }
            let result = try embedder.embed(audio: audioData)
            return Self.vector(from: result)
        }
    }

    /// `SearchQuery` is the right task for text -> audio retrieval and zero-shot labels.
    func embed(query text: String) throws -> [Float] {
        let input = addTaskPrefix ? "task: search result | query: \(text)" : text
        return try queue.sync {
            Self.vector(from: try embedder.embed(text: input))
        }
    }

    private static func vector(from result: EmbeddingResult) -> [Float] {
        guard let e = result.embeddings.first, let f = e.floatEmbedding else { return [] }
        return f.map { $0.floatValue }
    }
}

enum VectorMath {
    static func normalize(_ v: [Float]) -> [Float] {
        let n = sqrt(v.reduce(0) { $0 + $1 * $1 })
        return n > 0 ? v.map { $0 / n } : v
    }
    static func dot(_ a: [Float], _ b: [Float]) -> Float {
        var s: Float = 0
        for i in 0..<min(a.count, b.count) { s += a[i] * b[i] }
        return s
    }
    static func mean(_ vs: [[Float]]) -> [Float] {
        guard let first = vs.first else { return [] }
        var acc = [Float](repeating: 0, count: first.count)
        for v in vs { for i in 0..<acc.count { acc[i] += v[i] } }
        return normalize(acc.map { $0 / Float(vs.count) })
    }
}
