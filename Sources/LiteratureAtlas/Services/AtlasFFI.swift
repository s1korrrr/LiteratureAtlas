import Foundation
import Darwin
import AtlasFFIClib

// Runtime dynamic loader for atlas_ffi (Rust) — safe to fail and fallback to Swift search.
@available(macOS 26, iOS 26, *)
enum AtlasFFI {
    nonisolated(unsafe) private static let handle: UnsafeMutableRawPointer? = {
        let paths = [
            "analytics/ffi/target/release/libatlas_ffi.dylib",
            "./libatlas_ffi.dylib"
        ]
        for path in paths {
            if FileManager.default.fileExists(atPath: path), let h = dlopen(path, RTLD_NOW | RTLD_LOCAL) {
                return h
            }
        }
        return nil
    }()

    private static func symbol<T>(_ name: String, as type: T.Type) -> T? {
        guard let h = handle, let sym = dlsym(h, name) else { return nil }
        return unsafeBitCast(sym, to: type)
    }

    typealias BuildFn = @convention(c) (UInt32, UInt32, UnsafePointer<Float>?) -> UnsafeMutableRawPointer?
    typealias FreeFn = @convention(c) (UnsafeMutableRawPointer?) -> Void
    // Use opaque out pointer to avoid ObjC representability issues.
    typealias QueryFn = @convention(c) (UnsafeMutableRawPointer?, UnsafePointer<Float>?, UInt32, UnsafeMutableRawPointer?) -> UInt32

    static func isAvailable() -> Bool {
        #if ATLAS_FFI_LINKED
        return true
        #else
        return handle != nil
        #endif
    }

    static func buildIndex(vectors: [[Float]]) -> UnsafeMutableRawPointer? {
        guard let first = vectors.first, !first.isEmpty else { return nil }
        let dimCount = first.count
        guard vectors.allSatisfy({ $0.count == dimCount }) else { return nil }
        let dim = UInt32(dimCount)
        let n = UInt32(vectors.count)
        var flat: [Float] = []
        flat.reserveCapacity(dimCount * vectors.count)
        for v in vectors { flat.append(contentsOf: v) }

        #if ATLAS_FFI_LINKED
        return flat.withUnsafeBufferPointer { buf in
            atlas_build_index(dim, n, buf.baseAddress)
        }
        #else
        guard let build: BuildFn = symbol("atlas_build_index", as: BuildFn.self) else { return nil }
        return flat.withUnsafeBufferPointer { buf in
            build(dim, n, buf.baseAddress)
        }
        #endif
    }

    static func query(index: UnsafeMutableRawPointer?, query: [Float], k: Int) -> [AtlasSearchResult] {
        guard index != nil, k > 0, !query.isEmpty else { return [] }
        #if ATLAS_FFI_LINKED
        var results = [AtlasFFIClib.AtlasSearchResult](repeating: AtlasFFIClib.AtlasSearchResult(index: 0, distance: 0), count: k)
        let wrote = query.withUnsafeBufferPointer { qbuf -> UInt32 in
            results.withUnsafeMutableBufferPointer { outBuf -> UInt32 in
                atlas_query_index(index, qbuf.baseAddress, UInt32(k), outBuf.baseAddress)
            }
        }
        return Array(results.prefix(Int(wrote))).map { AtlasSearchResult(index: $0.index, distance: $0.distance) }
        #else
        guard let qfn: QueryFn = symbol("atlas_query_index", as: QueryFn.self) else { return [] }
        var results = [AtlasSearchResult](repeating: AtlasSearchResult(index: 0, distance: .infinity), count: k)
        let wrote = query.withUnsafeBufferPointer { qbuf -> UInt32 in
            results.withUnsafeMutableBytes { outBuf -> UInt32 in
                qfn(index, qbuf.baseAddress, UInt32(k), outBuf.baseAddress)
            }
        }
        return Array(results.prefix(Int(wrote)))
        #endif
    }

    static func free(index: UnsafeMutableRawPointer?) {
        #if ATLAS_FFI_LINKED
        atlas_free_index(index)
        #else
        guard let f: FreeFn = symbol("atlas_free_index", as: FreeFn.self) else { return }
        f(index)
        #endif
    }
}

struct AtlasSearchResult: Equatable {
    let index: UInt32
    let distance: Float
}
