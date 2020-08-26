load("@npm_bazel_typescript//:index.bzl", "ts_library")

package(default_visibility = ["//frontend:internal"])

ts_library(
    name = "memory_usage",
    srcs = [
        "memory_usage.ts",
    ],
    deps = [
        "@org_xprof//frontend/app/common/interfaces",
        "@org_xprof//frontend/app/common/interfaces:xla_proto_defs",
        "@org_xprof//frontend/app/common/utils",
        "@org_xprof//frontend/app/components/memory_viewer/xla",
    ],
)


