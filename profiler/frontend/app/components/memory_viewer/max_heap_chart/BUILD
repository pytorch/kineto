load("@npm_angular_bazel//:index.bzl", "ng_module")
load("@io_bazel_rules_sass//:defs.bzl", "sass_binary")

package(default_visibility = ["//frontend:internal"])

ng_module(
    name = "max_heap_chart",
    srcs = [
        "max_heap_chart.ts",
        "max_heap_chart_module.ts",
    ],
    assets = [
        ":max_heap_chart_css",
        "max_heap_chart.ng.html",
    ],
    deps = [
        "@npm//@angular/core",
        "@org_xprof//frontend/third_party/google_visualization",
        "@org_xprof//frontend/app/common/interfaces",
        "@org_xprof//frontend/app/common/utils",
    ],
)



sass_binary(
    name = "max_heap_chart_css",
    src = "max_heap_chart.scss",

    sourcemap = False,
    deps = [
        "@org_xprof//frontend/app/styles:common",
    ],
)
