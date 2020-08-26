load("@npm_angular_bazel//:index.bzl", "ng_module")
load("@io_bazel_rules_sass//:defs.bzl", "sass_binary")

package(default_visibility = ["//frontend:internal"])

ng_module(
    name = "program_order_chart",
    srcs = [
        "program_order_chart.ts",
        "program_order_chart_module.ts",
    ],
    assets = [
        ":program_order_chart_css",
        "program_order_chart.ng.html",
    ],
    deps = [
        "@npm//@angular/core",
        "@org_xprof//frontend/third_party/google_visualization",
        "@org_xprof//frontend/app/common/interfaces",
    ],
)



sass_binary(
    name = "program_order_chart_css",
    src = "program_order_chart.scss",

    sourcemap = False,
)
