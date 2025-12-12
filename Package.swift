// swift-tools-version: 5.9
// The swift-tools-version declares the minimum version of Swift required to build this package.

import PackageDescription

let package = Package(
    name: "neucodec-mlx-swift",
    platforms: [.macOS(.v14), .iOS(.v16)],
    products: [
        .library(name: "NeuCodec", targets: ["NeuCodec"])
    ],
    dependencies: [
        .package(url: "https://github.com/ml-explore/mlx-swift", from: "0.29.1"),
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
