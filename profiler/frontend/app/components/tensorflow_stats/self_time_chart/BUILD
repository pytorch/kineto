load("@npm_angular_bazel//:index.bzl", "ng_module")
load("@io_bazel_rules_sass//:defs.bzl", "sass_binary")

package(default_visibility = ["//frontend:internal"])

ng_module(
    name = "self_time_chart",
    srcs = [
        "self_time_chart.ts",
        "self_time_chart_module.ts",
    ],
    assets = [
        ":self_time_chart_css",
        "self_time_chart.ng.html",
    ],
    deps = [
        "@npm//@angular/core",
        "@org_xprof//frontend/app/common/angular:angular_material_card",
        "@org_xprof//frontend/third_party/google_visualization",
        "@org_xprof//frontend/app/common/constants",
        "@org_xprof//frontend/app/common/interfaces",
    ],
)



sass_binary(
    name = "self_time_chart_css",
    src = "self_time_chart.scss",

    sourcemap = False,
    deps = [
        "@org_xprof//frontend/app/styles:common",
    ],
)
