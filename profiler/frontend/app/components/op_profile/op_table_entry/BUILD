load("@npm_angular_bazel//:index.bzl", "ng_module")
load("@io_bazel_rules_sass//:defs.bzl", "sass_binary")

package(default_visibility = ["//frontend:internal"])

ng_module(
    name = "op_table_entry",
    srcs = [
        "op_table_entry.ts",
        "op_table_entry_module.ts",
    ],
    assets = [
        ":op_table_entry_css",
        "op_table_entry.ng.html",
    ],
    deps = [
        "@npm//@angular/common",
        "@npm//@angular/core",
        "@org_xprof//frontend/app/common/interfaces:op_profile_proto_defs",
        "@org_xprof//frontend/app/common/utils",
    ],
)



sass_binary(
    name = "op_table_entry_css",
    src = "op_table_entry.scss",

    sourcemap = False,
    deps = [
        "@org_xprof//frontend/app/styles:common",
    ],
)
