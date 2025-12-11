import SwiftUI

@available(macOS 26, iOS 26, *)
struct RootView: View {
    var body: some View {
        ZStack {
            LinearGradient(
                colors: [Color.blue.opacity(0.16), Color.purple.opacity(0.14), Color.indigo.opacity(0.12)],
                startPoint: .topLeading,
                endPoint: .bottomTrailing
            )
            .ignoresSafeArea()

            TabView {
                IngestView()
                    .tabItem { Label("Ingest", systemImage: "tray.and.arrow.down") }

                MapView()
                    .tabItem { Label("Map", systemImage: "circle.grid.3x3") }

                QuestionView()
                    .tabItem { Label("Q&A", systemImage: "questionmark.circle") }

                AnalyticsView()
                    .tabItem { Label("Analytics", systemImage: "chart.xyaxis.line") }
            }

            GlobalProgressOverlay()
        }
    }
}
