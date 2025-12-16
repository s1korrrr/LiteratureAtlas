import SwiftUI

private struct ViewWidthPreferenceKey: PreferenceKey {
    static var defaultValue: CGFloat { 0 }

    static func reduce(value: inout CGFloat, nextValue: () -> CGFloat) {
        let next = nextValue()
        if next > 0 {
            value = next
        }
    }
}

extension View {
    func onWidthChange(_ action: @escaping (CGFloat) -> Void) -> some View {
        background(
            GeometryReader { proxy in
                Color.clear.preference(key: ViewWidthPreferenceKey.self, value: proxy.size.width)
            }
        )
        .onPreferenceChange(ViewWidthPreferenceKey.self, perform: action)
    }
}
