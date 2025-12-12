import SwiftUI

@main
struct LiteratureAtlasApp: App {
    var body: some Scene {
        WindowGroup {
            if #available(macOS 26, iOS 26, *) {
                RootView()
                    .environmentObject(AppModel())
            } else {
                UnsupportedView()
            }
        }
    }
}

struct UnsupportedView: View {
    var body: some View {
        VStack(spacing: 12) {
            Text("Requires Apple Intelligence")
                .font(.title2.bold())
            Text("This app needs macOS 26 or iOS/iPadOS 26 with on-device Foundation Models and NLContextualEmbedding.")
                .multilineTextAlignment(.center)
                .foregroundStyle(.secondary)
        }
        .padding()
    }
}
