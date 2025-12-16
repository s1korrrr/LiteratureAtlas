import SwiftUI

@available(macOS 26, iOS 26, *)
struct DismissibleOverlay<Content: View>: View {
    let onDismiss: () -> Void
    @ViewBuilder let content: () -> Content

    @Environment(\.accessibilityReduceMotion) private var reduceMotion
    @State private var isVisible: Bool = false

    var body: some View {
        ZStack {
            Color.black
                .opacity(isVisible ? 0.46 : 0.0)
                .ignoresSafeArea()
                .onTapGesture { onDismiss() }

            content()
                .contentShape(Rectangle())
                .transition(reduceMotion ? .opacity : .opacity.combined(with: .scale(scale: 0.985)))
                .shadow(color: .black.opacity(0.35), radius: 26, x: 0, y: 18)
        }
        .onAppear { isVisible = true }
        .onExitCommand { onDismiss() }
        .animation(reduceMotion ? nil : .spring(response: 0.35, dampingFraction: 0.9), value: isVisible)
    }
}
