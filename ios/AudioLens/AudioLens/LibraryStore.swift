import Foundation
import SwiftUI

struct ClipWindow: Codable, Identifiable {
    var id: String { "\(fileID)-\(Int(start))" }
    let fileID: String
    let start: Double
    let end: Double
    var vector: [Float]
}

struct AudioClip: Codable, Identifiable {
    let id: String              // file name inside Documents/Audio
    let displayName: String
    let duration: Double
    var fileVector: [Float]     // normalised mean of window vectors
    var windowCount: Int
}

struct Hit: Identifiable {
    var id: String { "\(clip.id)-\(start)" }
    let clip: AudioClip
    let score: Float
    let start: Double
    let end: Double
}

struct LabelResult: Identifiable {
    var id: String { clip.id }
    let clip: AudioClip
    let label: String
    let confidence: Float
    let scores: [(String, Float)]
}

@MainActor
final class LibraryStore: ObservableObject {
    @Published var clips: [AudioClip] = []
    @Published var windows: [ClipWindow] = []
    @Published var status = "Idle"
    @Published var isBusy = false
    @Published var modelReady = false
    @Published var lastError: String?
    @AppStorage("windowSeconds") var windowSeconds = 60.0
    @AppStorage("useGPU") var useGPU = false

    private var service: EmbeddingService?
    private let fm = FileManager.default

    var audioDir: URL {
        let d = fm.urls(for: .documentDirectory, in: .userDomainMask)[0].appendingPathComponent("Audio")
        try? fm.createDirectory(at: d, withIntermediateDirectories: true)
        return d
    }
    private var indexURL: URL {
        fm.urls(for: .documentDirectory, in: .userDomainMask)[0].appendingPathComponent("index.json")
    }

    init() { load() }

    // MARK: model

    func loadModel() {
        guard service == nil else { modelReady = true; return }
        isBusy = true; status = "Loading EmbeddingGemma 2…"
        let gpu = useGPU
        Task.detached { [weak self] in
            do {
                let s = try EmbeddingService(useGPU: gpu)
                await MainActor.run { self?.service = s; self?.modelReady = true; self?.status = "Model ready"; self?.isBusy = false }
            } catch {
                await MainActor.run { self?.lastError = error.localizedDescription; self?.status = "Model failed"; self?.isBusy = false }
            }
        }
    }

    // MARK: import + index

    /// Copies picked files into Documents/Audio, then embeds anything not yet indexed.
    func importFiles(_ urls: [URL]) {
        for url in urls {
            let scoped = url.startAccessingSecurityScopedResource()
            defer { if scoped { url.stopAccessingSecurityScopedResource() } }
            let dest = audioDir.appendingPathComponent(url.lastPathComponent)
            if !fm.fileExists(atPath: dest.path) { try? fm.copyItem(at: url, to: dest) }
        }
        indexPending()
    }

    func indexPending() {
        guard let service else { lastError = "Load the model first."; return }
        let known = Set(clips.map(\.id))
        let files = ((try? fm.contentsOfDirectory(at: audioDir, includingPropertiesForKeys: nil)) ?? [])
            .filter { !known.contains($0.lastPathComponent) && !$0.lastPathComponent.hasPrefix(".") }
        guard !files.isEmpty else { status = "Nothing new to index"; return }
        isBusy = true
        let windowSeconds = self.windowSeconds
        Task.detached { [weak self] in
            for (i, url) in files.enumerated() {
                await MainActor.run { self?.status = "Embedding \(i + 1)/\(files.count): \(url.lastPathComponent)" }
                do {
                    let samples = try AudioDecoder.decode(url: url)
                    guard samples.count >= 4000 else { continue }
                    let wins = AudioDecoder.windows(samples, windowSeconds: windowSeconds)
                    var vecs: [ClipWindow] = []
                    for w in wins {
                        let v = try service.embed(audio: w.samples)
                        vecs.append(ClipWindow(fileID: url.lastPathComponent, start: w.start, end: w.end, vector: v))
                    }
                    let clip = AudioClip(id: url.lastPathComponent, displayName: url.lastPathComponent,
                                         duration: Double(samples.count) / AudioDecoder.sampleRate,
                                         fileVector: VectorMath.mean(vecs.map(\.vector)), windowCount: vecs.count)
                    await MainActor.run {
                        self?.clips.append(clip); self?.windows.append(contentsOf: vecs); self?.save()
                    }
                } catch {
                    await MainActor.run { self?.lastError = "\(url.lastPathComponent): \(error.localizedDescription)" }
                }
            }
            await MainActor.run { self?.status = "Indexed \(self?.clips.count ?? 0) files"; self?.isBusy = false }
        }
    }

    func remove(_ clip: AudioClip) {
        clips.removeAll { $0.id == clip.id }
        windows.removeAll { $0.fileID == clip.id }
        try? fm.removeItem(at: audioDir.appendingPathComponent(clip.id))
        save()
    }

    // MARK: analysis

    func search(_ text: String, k: Int = 20, perFile: Bool = true) -> [Hit] {
        guard let service, let q = try? service.embed(query: text) else { return [] }
        return rank(query: q, k: k, perFile: perFile)
    }

    func similar(to clip: AudioClip, k: Int = 20) -> [Hit] {
        rank(query: clip.fileVector, k: k, perFile: true).filter { $0.clip.id != clip.id }
    }

    private func rank(query q: [Float], k: Int, perFile: Bool) -> [Hit] {
        let byID = Dictionary(uniqueKeysWithValues: clips.map { ($0.id, $0) })
        let scored = windows.map { ($0, VectorMath.dot($0.vector, q)) }.sorted { $0.1 > $1.1 }
        var hits: [Hit] = []; var seen = Set<String>()
        for (w, s) in scored {
            if perFile { if seen.contains(w.fileID) { continue }; seen.insert(w.fileID) }
            if let c = byID[w.fileID] { hits.append(Hit(clip: c, score: s, start: w.start, end: w.end)) }
            if hits.count >= k { break }
        }
        return hits
    }

    /// Zero-shot: each label becomes a text query; a file gets the label with the highest cosine.
    func classify(labels: [String], temperature: Float = 0.05) -> [LabelResult] {
        guard let service else { return [] }
        let L = labels.compactMap { try? service.embed(query: $0) }
        guard L.count == labels.count else { return [] }
        return clips.map { clip in
            let s = L.map { VectorMath.dot(clip.fileVector, $0) }
            let e = s.map { expf($0 / temperature) }; let z = e.reduce(0, +)
            let order = s.indices.sorted { s[$0] > s[$1] }
            return LabelResult(clip: clip, label: labels[order[0]], confidence: e[order[0]] / z,
                               scores: order.map { (labels[$0], s[$0]) })
        }
    }

    func duplicates(threshold: Float = 0.95) -> [(AudioClip, AudioClip, Float)] {
        var out: [(AudioClip, AudioClip, Float)] = []
        for i in 0..<clips.count {
            for j in (i + 1)..<clips.count {
                let s = VectorMath.dot(clips[i].fileVector, clips[j].fileVector)
                if s >= threshold { out.append((clips[i], clips[j], s)) }
            }
        }
        return out.sorted { $0.2 > $1.2 }
    }

    // MARK: persistence

    private struct Snapshot: Codable { var clips: [AudioClip]; var windows: [ClipWindow] }

    func save() {
        if let d = try? JSONEncoder().encode(Snapshot(clips: clips, windows: windows)) { try? d.write(to: indexURL) }
    }
    private func load() {
        if let d = try? Data(contentsOf: indexURL), let s = try? JSONDecoder().decode(Snapshot.self, from: d) {
            clips = s.clips; windows = s.windows
        }
    }
}
