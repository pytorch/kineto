load("@npm_angular_bazel//:index.bzl", "ng_module")
load("@io_bazel_rules_sass//:defs.bzl", "sass_binary")

package(default_visibility = ["//frontend:internal"])

licenses(["notice"])

ng_module(
    name = "recommendation_result_view",
    srcs = [
        "recommendation_result_view.ts",
        "recommendation_result_view_common.ts",
        "recommendation_result_view_module.ts",
    ],
    assets = [
        ":recommendation_result_view_css",
        "recommendation_result_view.ng.html",
    ],
    deps = [
        "@npm//@angular/common",
        "@npm//@angular/core",
        "@org_xprof//frontend/app/common/angular:angular_material_card",
        "@npm//@ngrx/store",
        "@org_xprof//frontend/app/common/interfaces",
        "@org_xprof//frontend/app/common/utils",
        "@org_xprof//frontend/app/store",
    ],
)



sass_binary(
    name = "recommendation_result_view_css",
    src = "recommendation_result_view.scss",

    sourcemap = False,
    deps = [
        "@org_xprof//frontend/app/styles:common",
    ],
)

filegroup(
    name = "html",
    srcs = [
        "recommendation_result_view.ng.html",
    ],
)

filegroup(
    name = "scss",
    srcs = [
        "recommendation_result_view.scss",
    ],
)
