load("@npm_angular_bazel//:index.bzl", "ng_module")
load("@io_bazel_rules_sass//:defs.bzl", "sass_binary")

package(default_visibility = ["//frontend:internal"])

ng_module(
    name = "trace_viewer",
    srcs = [
        "trace_viewer.ts",
        "trace_viewer_module.ts",
    ],
    assets = [
        ":trace_viewer_css",
        "trace_viewer.ng.html",
    ],
    deps = [
        "@npm//@angular/common",
        "@org_xprof//frontend/app/common/angular:angular_common_http",
        "@npm//@angular/core",
        "@npm//@angular/router",
        "@org_xprof//frontend/app/common/constants",
        "@org_xprof//frontend/app/common/interfaces",
        "@org_xprof//frontend/app/pipes",
    ],
)



sass_binary(
    name = "trace_viewer_css",
    src = "trace_viewer.scss",

    sourcemap = False,
)
