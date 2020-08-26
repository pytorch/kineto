load("@npm_angular_bazel//:index.bzl", "ng_module")
load("@io_bazel_rules_sass//:defs.bzl", "sass_binary")

package(default_visibility = ["//frontend:internal"])

ng_module(
    name = "diagnostics_view",
    srcs = [
        "diagnostics_view.ts",
        "diagnostics_view_module.ts",
    ],
    assets = [
        ":diagnostics_view_css",
        "diagnostics_view.ng.html",
    ],
    deps = [
        "@npm//@angular/core",
        "@org_xprof//frontend/app/common/angular:angular_material_card",
        "@org_xprof//frontend/app/common/interfaces",
    ],
)



sass_binary(
    name = "diagnostics_view_css",
    src = "diagnostics_view.scss",

    sourcemap = False,
    deps = [
        "@org_xprof//frontend/app/styles:common",
    ],
)
