import Foundation
import SwiftUI

@available(macOS 26, iOS 26, *)
@MainActor
final class AppNavigation: ObservableObject {
    enum Tab: Hashable {
        case ingest
        case map
        case qa
        case trading
        case projects
        case analytics
    }

    @Published var selectedTab: Tab = .ingest
    @Published var requestedStrategyProjectID: UUID? = nil
}
