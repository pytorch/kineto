load("@npm_angular_bazel//:index.bzl", "ng_module")
load("@io_bazel_rules_sass//:defs.bzl", "sass_binary")

package(default_visibility = ["//frontend:internal"])

ng_module(
    name = "operations_table",
    srcs = [
        "operations_table.ts",
        "operations_table_module.ts",
    ],
    assets = [
        ":operations_table_css",
        "operations_table.ng.html",
    ],
    deps = [
        "@npm//@angular/core",
        "@org_xprof//frontend/third_party/google_visualization",
        "@org_xprof//frontend/app/common/constants",
        "@org_xprof//frontend/app/common/interfaces",
    ],
)



sass_binary(
    name = "operations_table_css",
    src = "operations_table.scss",

    sourcemap = False,
    deps = [
        "@org_xprof//frontend/app/styles:common",
    ],
)
