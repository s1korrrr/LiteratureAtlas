import Foundation

enum ObsidianIDs {
    static func paperAlias(_ id: UUID) -> String {
        "paper-\(id.uuidString)"
    }

    static func clusterAlias(_ id: Int) -> String {
        "cluster-\(id)"
    }

    static func strategyAlias(_ id: UUID) -> String {
        "strategy-\(id.uuidString)"
    }
}

