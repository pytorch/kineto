load("@npm_bazel_typescript//:index.bzl", "ts_library")

package(default_visibility = ["//frontend:internal"])

ts_library(
    name = "utils",
    srcs = [
        "utils.ts",
    ],
    deps = [
        "@npm//@ngrx/store",
        "@org_xprof//frontend/app/common/interfaces",
        "@org_xprof//frontend/app/store",
    ],
)


