import SwiftUI

@main
struct AudioLensApp: App {
    @StateObject private var store = LibraryStore()

    var body: some Scene {
        WindowGroup {
            ContentView()
                .environmentObject(store)
        }
    }
}
