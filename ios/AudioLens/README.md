# AudioLens — EmbeddingGemma 2 audio analysis on iPhone

A small SwiftUI app that runs **EmbeddingGemma 2 (740M, LiteRT bundle)** fully on-device
through MediaPipe's `UniversalEmbedder`, and uses it to analyse your own recordings:

| Tab | What it does |
| --- | --- |
| Library | Import audio/video files (Files app, iCloud, AirDrop), index them in 60 s windows |
| Search | Natural-language search ("dog barking", "someone laughing"), tap a hit to play from that timestamp |
| Library ▸ file | "Similar" — audio-to-audio search |
| Classify | Zero-shot labels from any comma-separated list you type |
| Duplicates | Near-duplicate recordings by cosine similarity |

Everything stays on the phone. No server, no transcription step.

## What you need

* An iPhone on iOS 17+ (A14 or newer is comfortable; the model uses ~100–230 MB RAM).
* **A Mac with Xcode 16+ to build and install the app.** iOS apps can't be built on the phone
  itself. A free Apple ID is enough to run it on your own device (re-sign every 7 days);
  a paid developer account removes that limit.
* The model file `embeddinggemma-2-740m.litertlm` (485 MB) from
  https://huggingface.co/litert-community/embeddinggemma-2-740m-litert-lm
  (accept the Gemma licence with a Hugging Face account, then download from Files and versions).

## Build

```bash
cd ios/AudioLens
brew install xcodegen          # once
xcodegen generate              # creates AudioLens.xcodeproj from project.yml
open AudioLens.xcodeproj
```

Then in Xcode: select the AudioLens target ▸ Signing & Capabilities ▸ pick your Team,
plug in the iPhone, choose it as the run destination and press Run. Xcode resolves the
`MediaPipeTasksRetrieval` Swift package (MediaPipe ≥ 1.1.0) on first build.

Without XcodeGen: File ▸ New ▸ Project ▸ iOS App (SwiftUI), name it AudioLens, delete the
generated ContentView.swift / App file, drag the five `.swift` files from `AudioLens/` in,
add the package `https://github.com/google-ai-edge/mediapipe.git` (product
MediaPipeTasksRetrieval), and add `-ObjC` to Other Linker Flags.

## Put the model on the phone

Either:

* **Documents folder (no rebuild needed):** run the app once, then in the Files app go to
  On My iPhone ▸ AudioLens and drop `embeddinggemma-2-740m.litertlm` there (AirDrop from a Mac
  or download on the phone and Move). Relaunch.
* **Bundle it:** drag the file into the Xcode project with "Copy items" and the AudioLens
  target ticked. Adds 485 MB to the app.

## How it maps to the model card

* Audio is decoded with AVFoundation to **16 kHz mono Float32** and cut into windows
  (default 60 s, max 300 s — the 8192-token context holds ~327 s at 25 tokens/s).
* Audio is embedded with **no prompt**; text queries and labels get the
  `task: search result | query:` prefix (asymmetric text→audio retrieval).
  If you find MediaPipe already adds a prefix, set `addTaskPrefix = false` in
  `EmbeddingService.swift`.
* Vectors are L2-normalised (`l2Normalize = true`), so scores are cosine similarities.
* A file's vector is the normalised mean of its window vectors; search ranks windows so
  you get timestamps.
* Defaults to the CPU backend. Flip "Use GPU (Metal)" before the model loads for ~3× faster
  embedding on recent iPhones.

## Status / caveats

* Written against the MediaPipe 1.1.0 iOS docs published 2026-10-06 and **not yet compiled**
  (no Mac in the environment this was authored in). Expect at most small type fixes in
  `EmbeddingService.swift` — e.g. `Int` vs `UInt` on `FloatBuffer`/`AudioData`, or the
  `delegate` enum spelling — Xcode will point straight at them.
* If `AudioData` / `AudioDataFormat` / `FloatBuffer` aren't found, also add the
  `MediaPipeTasksAudio` product from the same package.
* The index is a JSON file in Documents; fine for hundreds of files. For thousands, swap in
  MediaPipe's `SemanticRetriever` + `SqliteVectorStore`, which the same package provides.
* Long files are embedded window by window; a 1-hour recording is ~60 model calls
  (roughly 0.2–0.5 s each on an iPhone 15/16 class device).
