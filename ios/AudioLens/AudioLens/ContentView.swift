import SwiftUI
import AVFoundation
import UniformTypeIdentifiers

struct ContentView: View {
    @EnvironmentObject var store: LibraryStore

    var body: some View {
        TabView {
            LibraryView().tabItem { Label("Library", systemImage: "waveform") }
            SearchView().tabItem { Label("Search", systemImage: "magnifyingglass") }
            ClassifyView().tabItem { Label("Classify", systemImage: "tag") }
            DuplicatesView().tabItem { Label("Duplicates", systemImage: "doc.on.doc") }
        }
        .onAppear { store.loadModel() }
        .alert("Error", isPresented: Binding(get: { store.lastError != nil }, set: { _ in store.lastError = nil })) {
            Button("OK") {}
        } message: { Text(store.lastError ?? "") }
    }
}

struct LibraryView: View {
    @EnvironmentObject var store: LibraryStore
    @State private var showPicker = false

    var body: some View {
        NavigationStack {
            List {
                Section {
                    HStack {
                        if store.isBusy { ProgressView().padding(.trailing, 6) }
                        Text(store.status).font(.footnote).foregroundStyle(.secondary)
                    }
                    Toggle("Use GPU (Metal)", isOn: $store.useGPU).disabled(store.modelReady)
                    Stepper("Window: \(Int(store.windowSeconds)) s", value: $store.windowSeconds, in: 10...300, step: 10)
                }
                Section("\(store.clips.count) files") {
                    ForEach(store.clips) { clip in
                        NavigationLink {
                            SimilarView(clip: clip)
                        } label: {
                            VStack(alignment: .leading) {
                                Text(clip.displayName).lineLimit(1)
                                Text("\(fmt(clip.duration)) · \(clip.windowCount) window(s)")
                                    .font(.caption).foregroundStyle(.secondary)
                            }
                        }
                    }
                    .onDelete { idx in idx.map { store.clips[$0] }.forEach(store.remove) }
                }
            }
            .navigationTitle("AudioLens")
            .toolbar {
                Button { showPicker = true } label: { Label("Import", systemImage: "plus") }
                    .disabled(!store.modelReady || store.isBusy)
                Button { store.indexPending() } label: { Label("Index new", systemImage: "arrow.clockwise") }
                    .disabled(!store.modelReady || store.isBusy)
            }
            .fileImporter(isPresented: $showPicker, allowedContentTypes: [.audio, .movie, .mpeg4Movie],
                          allowsMultipleSelection: true) { result in
                if case .success(let urls) = result { store.importFiles(urls) }
            }
        }
    }
}

struct SearchView: View {
    @EnvironmentObject var store: LibraryStore
    @State private var query = ""
    @State private var perFile = true
    @State private var hits: [Hit] = []

    var body: some View {
        NavigationStack {
            VStack {
                HStack {
                    TextField("e.g. dog barking, someone laughing", text: $query)
                        .textFieldStyle(.roundedBorder)
                        .onSubmit(run)
                    Button("Go", action: run).disabled(query.isEmpty || !store.modelReady)
                }.padding(.horizontal)
                Toggle("One result per file", isOn: $perFile).padding(.horizontal).onChange(of: perFile) { _, _ in run() }
                HitList(hits: hits)
            }
            .navigationTitle("Search")
        }
    }
    private func run() { hits = store.search(query, perFile: perFile) }
}

struct SimilarView: View {
    @EnvironmentObject var store: LibraryStore
    let clip: AudioClip
    var body: some View {
        HitList(hits: store.similar(to: clip)).navigationTitle("Similar to \(clip.displayName)")
    }
}

struct HitList: View {
    @EnvironmentObject var store: LibraryStore
    let hits: [Hit]
    @StateObject private var player = Player()

    var body: some View {
        List(hits) { h in
            Button {
                player.play(url: store.audioDir.appendingPathComponent(h.clip.id), at: h.start)
            } label: {
                HStack {
                    Text(String(format: "%.3f", h.score)).monospacedDigit().foregroundStyle(.secondary)
                    VStack(alignment: .leading) {
                        Text(h.clip.displayName).lineLimit(1)
                        Text("\(fmt(h.start)) – \(fmt(h.end))").font(.caption).foregroundStyle(.secondary)
                    }
                    Spacer()
                    Image(systemName: "play.circle")
                }
            }
        }
        .overlay { if hits.isEmpty { Text("No results").foregroundStyle(.secondary) } }
    }
}

struct ClassifyView: View {
    @EnvironmentObject var store: LibraryStore
    @AppStorage("labels") private var labelText = "speech, music, traffic, birdsong, silence"
    @State private var results: [LabelResult] = []

    var body: some View {
        NavigationStack {
            VStack {
                TextField("Comma-separated labels", text: $labelText, axis: .vertical)
                    .textFieldStyle(.roundedBorder).padding(.horizontal)
                Button("Classify all files") {
                    let labels = labelText.split(separator: ",").map { $0.trimmingCharacters(in: .whitespaces) }.filter { !$0.isEmpty }
                    results = store.classify(labels: labels)
                }.disabled(!store.modelReady || store.clips.isEmpty)
                List(results) { r in
                    VStack(alignment: .leading) {
                        HStack {
                            Text(r.label).bold()
                            Text(String(format: "p=%.2f", r.confidence)).foregroundStyle(.secondary)
                        }
                        Text(r.clip.displayName).font(.caption).lineLimit(1)
                        Text(r.scores.prefix(3).map { String(format: "%@ %.3f", $0.0, $0.1) }.joined(separator: " · "))
                            .font(.caption2).foregroundStyle(.secondary)
                    }
                }
            }
            .navigationTitle("Zero-shot labels")
        }
    }
}

struct DuplicatesView: View {
    @EnvironmentObject var store: LibraryStore
    @State private var threshold: Float = 0.95
    var body: some View {
        NavigationStack {
            VStack {
                HStack {
                    Text(String(format: "Threshold %.2f", threshold))
                    Slider(value: $threshold, in: 0.7...0.999)
                }.padding(.horizontal)
                let pairs = store.duplicates(threshold: threshold)
                List(pairs.indices, id: \.self) { i in
                    let p = pairs[i]
                    VStack(alignment: .leading) {
                        Text(String(format: "%.4f", p.2)).monospacedDigit().foregroundStyle(.secondary)
                        Text(p.0.displayName).lineLimit(1)
                        Text(p.1.displayName).lineLimit(1)
                    }
                }
                .overlay { if pairs.isEmpty { Text("No near-duplicates").foregroundStyle(.secondary) } }
            }
            .navigationTitle("Duplicates")
        }
    }
}

final class Player: ObservableObject {
    private var player: AVAudioPlayer?
    func play(url: URL, at seconds: Double) {
        try? AVAudioSession.sharedInstance().setCategory(.playback)
        try? AVAudioSession.sharedInstance().setActive(true)
        player = try? AVAudioPlayer(contentsOf: url)
        player?.currentTime = seconds
        player?.play()
    }
}

func fmt(_ s: Double) -> String {
    let t = Int(s)
    return t >= 3600 ? String(format: "%d:%02d:%02d", t / 3600, t % 3600 / 60, t % 60)
                     : String(format: "%d:%02d", t / 60, t % 60)
}
