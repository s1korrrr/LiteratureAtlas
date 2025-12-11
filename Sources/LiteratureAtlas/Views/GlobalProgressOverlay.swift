import SwiftUI

@available(macOS 26, iOS 26, *)
struct GlobalProgressOverlay: View {
    @EnvironmentObject private var model: AppModel

    var body: some View {
        VStack {
            Spacer()
            if model.isIngesting || model.isClustering {
                GlassCard {
                    HStack(spacing: 12) {
                        ProgressView(value: progressValue)
                            .frame(width: 200)
                        VStack(alignment: .leading, spacing: 4) {
                            Text(statusText)
                                .font(.headline)
                            Text(subtitleText)
                                .font(.caption)
                                .foregroundStyle(.secondary)
                        }
                        Spacer(minLength: 10)
                        Button {
                            if model.isIngesting { model.cancelIngestion() }
                        } label: {
                            Label("Stop", systemImage: "stop.fill")
                        }
                        .buttonStyle(.bordered)
                        .tint(.red)
                        .disabled(!model.isIngesting)
                    }
                }
                .padding(.bottom, 16)
                .transition(.move(edge: .bottom).combined(with: .opacity))
                .animation(.easeInOut, value: model.isIngesting || model.isClustering)
            }
        }
        .padding(.horizontal, 16)
        .ignoresSafeArea(edges: .bottom)
    }

    private var progressValue: Double {
        if model.isIngesting { return model.ingestionProgress }
        if model.isClustering { return model.clusteringProgress }
        return 0
    }

    private var statusText: String {
        if model.isIngesting { return "Ingesting PDFs" }
        if model.isClustering { return "Clustering" }
        return "Idle"
    }

    private var subtitleText: String {
        if model.isIngesting {
            let current = model.ingestionCurrentFile.isEmpty ? "" : " · " + model.ingestionCurrentFile
            return "\(model.ingestionCompletedCount)/\(max(model.ingestionTotalCount, 1)) files\(current)"
        }
        if model.isClustering {
            return String(format: "%.0f%%", model.clusteringProgress * 100)
        }
        return ""
    }
}
