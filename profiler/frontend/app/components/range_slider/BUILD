load("@npm_angular_bazel//:index.bzl", "ng_module")
load("@io_bazel_rules_sass//:defs.bzl", "sass_binary")

package(default_visibility = ["//frontend:internal"])

ng_module(
    name = "range_slider",
    srcs = [
        "range_slider.ts",
        "range_slider_module.ts",
    ],
    assets = [
        ":range_slider_css",
        "range_slider.ng.html",
    ],
    deps = [
        "@npm//@angular/core",
        "@org_xprof//frontend/app/common/angular:angular_material_slider",
    ],
)



sass_binary(
    name = "range_slider_css",
    src = "range_slider.scss",

    sourcemap = False,
    deps = [
        "@org_xprof//frontend/app/styles:common",
    ],
)
