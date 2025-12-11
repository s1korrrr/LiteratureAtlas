// swift-tools-version: 6.0

import PackageDescription

let package = Package(
    name: "LiteratureAtlas",
    platforms: [
        .macOS(.v15),
        .iOS(.v18)
    ],
    products: [
        .executable(name: "LiteratureAtlas", targets: ["LiteratureAtlas"])
    ],
    targets: [
        .systemLibrary(
            name: "AtlasFFIClib",
            path: "analytics/ffi/include",
            pkgConfig: nil,
            providers: []
        ),
        .executableTarget(
            name: "LiteratureAtlas",
            dependencies: [
                .target(name: "AtlasFFIClib")
            ],
            path: "Sources/LiteratureAtlas",
            linkerSettings: [
                .unsafeFlags(["-Lanalytics/ffi/target/release", "-latlas_ffi"], .when(platforms: [.macOS, .iOS]))
            ]
        ),
        .testTarget(
            name: "LiteratureAtlasTests",
            dependencies: ["LiteratureAtlas"],
            path: "Tests/LiteratureAtlasTests"
        )
    ]
)
