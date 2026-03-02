// swift-tools-version: 5.9
// The swift-tools-version declares the minimum version of Swift required to build this package.

import PackageDescription

let package = Package(
    name: "neucodec-mlx-swift",
    platforms: [.macOS(.v14), .iOS("15.8")],
    products: [
        .library(name: "NeuCodec", targets: ["NeuCodec"])
    ],
    dependencies: [
        .package(url: "https://github.com/wisent-ai/mlx-swift", branch: "ios15-compat"),
        .package(path: "../swift-transformers-local"),
    ],
    targets: [
        .target(
            name: "NeuCodec",
            dependencies: [
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXNN", package: "mlx-swift"),
                .product(name: "MLXRandom", package: "mlx-swift"),
                .product(name: "Transformers", package: "swift-transformers-local"),
            ]
        ),
        .testTarget(
            name: "NeuCodecTests",
            dependencies: ["NeuCodec"],
            resources: [.copy("Resources")]
        )
    ]
)
