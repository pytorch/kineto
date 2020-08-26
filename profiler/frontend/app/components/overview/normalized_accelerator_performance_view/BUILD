load("@npm_angular_bazel//:index.bzl", "ng_module")
load("@io_bazel_rules_sass//:defs.bzl", "sass_binary")

package(default_visibility = ["//frontend:internal"])

ng_module(
    name = "normalized_accelerator_performance_view",
    srcs = [
        "normalized_accelerator_performance_view.ts",
        "normalized_accelerator_performance_view_module.ts",
    ],
    assets = [
        ":normalized_accelerator_performance_view_css",
        "normalized_accelerator_performance_view.ng.html",
    ],
    deps = [
        "@npm//@angular/common",
        "@npm//@angular/core",
        "@org_xprof//frontend/app/common/angular:angular_material_expansion",
        "@org_xprof//frontend/app/common/interfaces",
    ],
)



sass_binary(
    name = "normalized_accelerator_performance_view_css",
    src = "normalized_accelerator_performance_view.scss",

    sourcemap = False,
    deps = [
        "@org_xprof//frontend/app/styles:common",
    ],
)
