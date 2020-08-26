load("@npm_angular_bazel//:index.bzl", "ng_module")
load("@npm_bazel_typescript//:index.bzl", "ts_library")

package(default_visibility = ["//frontend:internal"])

ng_module(
    name = "chart",
    srcs = [
        "chart.ts",
    ],
    deps = [
        ":default_data_provider",
        "@org_xprof//frontend/third_party/google_visualization",
        "@org_xprof//frontend/app/common/interfaces",
        "@org_xprof//frontend/app/common/interfaces:chart",
        "@org_xprof//frontend/app/common/typing/google_visualization",
    ],
)

ts_library(
    name = "default_data_provider",
    srcs = [
        "default_data_provider.ts",
    ],
    deps = [
        "@org_xprof//frontend/third_party/google_visualization",
        "@org_xprof//frontend/app/common/interfaces:chart",
        "@org_xprof//frontend/app/common/typing/google_visualization",
    ],
)

ts_library(
    name = "chart_options",
    srcs = [
        "chart_options.ts",
    ],
)


