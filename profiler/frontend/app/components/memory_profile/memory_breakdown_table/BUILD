load("@npm_angular_bazel//:index.bzl", "ng_module")
load("@io_bazel_rules_sass//:defs.bzl", "sass_binary")

package(default_visibility = ["//frontend:internal"])

ng_module(
    name = "memory_breakdown_table",
    srcs = [
        "memory_breakdown_table.ts",
        "memory_breakdown_table_module.ts",
    ],
    assets = [
        ":memory_breakdown_table_css",
        "memory_breakdown_table.ng.html",
    ],
    deps = [
        "@npm//@angular/core",
        "@org_xprof//frontend/app/common/angular:angular_material_form_field",
        "@org_xprof//frontend/app/common/angular:angular_material_icon",
        "@org_xprof//frontend/app/common/angular:angular_material_input",
        "@org_xprof//frontend/third_party/google_visualization",
        "@org_xprof//frontend/app/common/interfaces",
        "@org_xprof//frontend/app/common/interfaces:memory_profile_proto_defs",
    ],
)



sass_binary(
    name = "memory_breakdown_table_css",
    src = "memory_breakdown_table.scss",

    sourcemap = False,
    deps = [
        "@org_xprof//frontend/app/styles:common",
    ],
)
