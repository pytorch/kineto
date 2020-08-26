load("@npm_angular_bazel//:index.bzl", "ng_module")
load("@io_bazel_rules_sass//:defs.bzl", "sass_binary")

package(default_visibility = ["//frontend:internal"])

ng_module(
    name = "kernel_stats_table",
    srcs = [
        "kernel_stats_table.ts",
        "kernel_stats_table_module.ts",
    ],
    assets = [
        ":kernel_stats_table_css",
        "kernel_stats_table.ng.html",
    ],
    deps = [
        "@npm//@angular/core",
        "@org_xprof//frontend/app/common/angular:angular_material_form_field",
        "@org_xprof//frontend/app/common/angular:angular_material_icon",
        "@org_xprof//frontend/app/common/angular:angular_material_input",
        "@org_xprof//frontend/third_party/google_visualization",
        "@org_xprof//frontend/app/common/interfaces",
    ],
)



sass_binary(
    name = "kernel_stats_table_css",
    src = "kernel_stats_table.scss",

    sourcemap = False,
    deps = [
        "@org_xprof//frontend/app/styles:common",
    ],
)
