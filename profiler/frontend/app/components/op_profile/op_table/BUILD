load("@npm_angular_bazel//:index.bzl", "ng_module")
load("@io_bazel_rules_sass//:defs.bzl", "sass_binary")

package(default_visibility = ["//frontend:internal"])

ng_module(
    name = "op_table",
    srcs = [
        "op_table.ts",
        "op_table_module.ts",
    ],
    assets = [
        ":op_table_css",
        "op_table.ng.html",
    ],
    deps = [
        "@npm//@angular/common",
        "@npm//@angular/core",
        "@npm//@ngrx/store",
        "@org_xprof//frontend/app/common/interfaces:op_profile_proto_defs",
        "@org_xprof//frontend/app/components/op_profile/op_table_entry",
        "@org_xprof//frontend/app/store",
    ],
)



sass_binary(
    name = "op_table_css",
    src = "op_table.scss",

    sourcemap = False,
)
