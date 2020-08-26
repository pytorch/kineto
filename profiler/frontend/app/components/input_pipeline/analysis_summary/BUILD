load("@npm_angular_bazel//:index.bzl", "ng_module")
load("@io_bazel_rules_sass//:defs.bzl", "sass_binary")

package(default_visibility = ["//frontend:internal"])

ng_module(
    name = "analysis_summary",
    srcs = [
        "analysis_summary.ts",
        "analysis_summary_module.ts",
    ],
    assets = [
        ":analysis_summary_css",
        "analysis_summary.ng.html",
    ],
    deps = [
        "@npm//@angular/core",
        "@org_xprof//frontend/app/common/interfaces",
    ],
)



sass_binary(
    name = "analysis_summary_css",
    src = "analysis_summary.scss",

    sourcemap = False,
    deps = [
        "@org_xprof//frontend/app/styles:common",
    ],
)
