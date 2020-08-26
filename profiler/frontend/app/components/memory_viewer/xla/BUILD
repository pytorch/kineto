load("@npm_bazel_typescript//:index.bzl", "ts_library")

package(default_visibility = ["//frontend:internal"])

ts_library(
    name = "xla",
    srcs = [
        "buffer_allocation.ts",
        "buffer_allocation_assigned.ts",
        "hlo_instruction.ts",
        "layout.ts",
        "logical_buffer.ts",
        "shape.ts",
    ],
    deps = [
        "@org_xprof//frontend/app/common/interfaces:xla_proto_defs",
        "@org_xprof//frontend/app/common/utils",
    ],
)


