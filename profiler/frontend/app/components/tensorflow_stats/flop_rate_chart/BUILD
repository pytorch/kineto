load("@npm_angular_bazel//:index.bzl", "ng_module")
load("@io_bazel_rules_sass//:defs.bzl", "sass_binary")

package(default_visibility = ["//frontend:internal"])

ng_module(
    name = "flop_rate_chart",
    srcs = [
        "flop_rate_chart.ts",
        "flop_rate_chart_module.ts",
    ],
    assets = [
        ":flop_rate_chart_css",
        "flop_rate_chart.ng.html",
    ],
    deps = [
        "@npm//@angular/core",
        "@org_xprof//frontend/app/common/angular:angular_material_card",
        "@org_xprof//frontend/third_party/google_visualization",
        "@org_xprof//frontend/app/common/interfaces",
    ],
)



sass_binary(
    name = "flop_rate_chart_css",
    src = "flop_rate_chart.scss",

    sourcemap = False,
)
