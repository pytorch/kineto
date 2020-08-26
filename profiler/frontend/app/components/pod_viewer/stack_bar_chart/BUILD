load("@npm_angular_bazel//:index.bzl", "ng_module")
load("@io_bazel_rules_sass//:defs.bzl", "sass_binary")

package(default_visibility = ["//frontend:internal"])

ng_module(
    name = "stack_bar_chart",
    srcs = [
        "stack_bar_chart.ts",
        "stack_bar_chart_module.ts",
    ],
    assets = [
        ":stack_bar_chart_css",
        "stack_bar_chart.ng.html",
    ],
    deps = [
        "@npm//@angular/core",
        "@org_xprof//frontend/third_party/google_visualization",
        "@org_xprof//frontend/app/common/interfaces",
    ],
)

sass_binary(
    name = "stack_bar_chart_css",
    src = "stack_bar_chart.scss",

    sourcemap = False,
)
