// swift-tools-version: 5.6
// The swift-tools-version declares the minimum version of Swift required to build this package.

import PackageDescription

#if os(macOS)

let package = Package(
    name: "llama.swift",
    platforms: [.macOS(.v11)],
    products: [
        // Products define the executables and libraries a package produces, and make them visible to other packages.
        .library(
            name: "CLLaMa",
            targets: ["CLLaMa"]),
    ],
    dependencies: [
        // Dependencies declare other packages that this package depends on.
        // .package(url: /* package url */, from: "1.0.0"),
    ],
    targets: [
        // Targets are the basic building blocks of a package. A target can define a module or a test suite.
        // Targets can depend on other targets in this package, and on products in packages this package depends on.
        .target(
            name: "CLLaMa",
            sources: ["ggml.c", "llama.cpp"],
            publicHeadersPath: "spm-headers",
            cSettings: [
                .unsafeFlags(["-O3"]),
                .unsafeFlags(["-DNDEBUG"]),
                .unsafeFlags(["-mfma"]),
                .unsafeFlags(["-mavx"]),
                .unsafeFlags(["-mavx2"]),
                .unsafeFlags(["-mf16c"]),
                .unsafeFlags(["-msse3"]),
                .unsafeFlags(["-DGGML_USE_ACCELERATE"]),
                .unsafeFlags(["-w"])    // ignore all warnings
            ]),
    ],
    cxxLanguageStandard: CXXLanguageStandard.cxx11
)

#endif

#if os(iOS)

let package = Package(
    name: "llama.swift",
    platforms: [.iOS(.v15)],
    products: [
        // Products define the executables and libraries a package produces, and make them visible to other packages.
        .library(
            name: "CLLaMa",
            targets: ["CLLaMa"]),
    ],
    dependencies: [
        // Dependencies declare other packages that this package depends on.
        // .package(url: /* package url */, from: "1.0.0"),
    ],
    targets: [
        // Targets are the basic building blocks of a package. A target can define a module or a test suite.
        // Targets can depend on other targets in this package, and on products in packages this package depends on.
        .target(
            name: "CLLaMa",
            sources: ["ggml.c", "llama.cpp", "gptneox.cpp"],
            publicHeadersPath: "spm-headers",
            cSettings: [
                .unsafeFlags(["-O3"]),
                .unsafeFlags(["-DNDEBUG"]),
                .unsafeFlags(["-mcpu=native"]),
                .unsafeFlags(["-DGGML_USE_ACCELERATE"]),
                .unsafeFlags(["-w"])    // ignore all warnings
            ]),
    ],
    cxxLanguageStandard: CXXLanguageStandard.cxx11
)

#endif
