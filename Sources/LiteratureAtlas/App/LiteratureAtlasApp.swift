import SwiftUI
import FoundationModels

#if os(macOS)
import AppKit

@available(macOS 26, *)
@MainActor
final class LiteratureAtlasAppDelegate: NSObject, NSApplicationDelegate {
    private func bringAllWindowsToFront() {
        NSApp.activate(ignoringOtherApps: true)
        for window in NSApp.windows {
            window.makeKeyAndOrderFront(nil)
        }
    }

    func applicationDidFinishLaunching(_ notification: Notification) {
        NSApp.setActivationPolicy(.regular)
        Task { @MainActor in
            self.bringAllWindowsToFront()
        }
    }

    func applicationDidBecomeActive(_ notification: Notification) {
        Task { @MainActor in
            self.bringAllWindowsToFront()
        }
    }
}
#endif

@main
struct LiteratureAtlasApp: App {
    @StateObject private var model = AppModel()

    #if os(macOS)
    @NSApplicationDelegateAdaptor(LiteratureAtlasAppDelegate.self) private var appDelegate
    #endif

    var body: some Scene {
        WindowGroup {
            let availability = SystemLanguageModel.default.availability
            switch availability {
            case .available:
                RootView()
                    .environmentObject(model)
            case .unavailable(let reason):
                UnsupportedView(reason: String(describing: reason))
            }
        }
    }
}

struct UnsupportedView: View {
    let reason: String

    var body: some View {
        VStack(spacing: 12) {
            Text("On-device model unavailable")
                .font(.title2.bold())
            Text("LiteratureAtlas requires an on-device Foundation Models language model for summaries and Q&A.")
                .multilineTextAlignment(.center)
                .foregroundStyle(.secondary)
            Text(reason)
                .font(.caption)
                .multilineTextAlignment(.center)
                .foregroundStyle(.secondary)
        }
        .padding()
    }
}
