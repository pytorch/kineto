load("@npm_angular_bazel//:index.bzl", "ng_module")
load("@io_bazel_rules_sass//:defs.bzl", "sass_binary")

package(default_visibility = ["//frontend:internal"])

ng_module(
    name = "run_environment_view",
    srcs = [
        "run_environment_view.ts",
        "run_environment_view_module.ts",
    ],
    assets = [
        ":run_environment_view_css",
        "run_environment_view.ng.html",
    ],
    deps = [
        "@npm//@angular/core",
        "@org_xprof//frontend/app/common/angular:angular_material_card",
        "@org_xprof//frontend/app/common/interfaces",
    ],
)



sass_binary(
    name = "run_environment_view_css",
    src = "run_environment_view.scss",

    sourcemap = False,
    deps = [
        "@org_xprof//frontend/app/styles:common",
    ],
)
