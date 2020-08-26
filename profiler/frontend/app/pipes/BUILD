load("@npm_angular_bazel//:index.bzl", "ng_module")

package(default_visibility = ["//frontend:internal"])

ng_module(
    name = "pipes",
    srcs = [
        "pipes_module.ts",
        "safe_pipe.ts",
    ],
    deps = [
        "@npm//@angular/common",
        "@npm//@angular/core",
        "@npm//@angular/platform-browser",
    ],
)
