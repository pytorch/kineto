load("@npm_angular_bazel//:index.bzl", "ng_module")
load("@io_bazel_rules_sass//:defs.bzl", "sass_binary")

package(default_visibility = ["//frontend:internal"])

ng_module(
    name = "pod_viewer_details",
    srcs = [
        "pod_viewer_details.ts",
        "pod_viewer_details_module.ts",
    ],
    assets = [
        ":pod_viewer_details_css",
        "pod_viewer_details.ng.html",
    ],
    deps = [
        "@npm//@angular/common",
        "@npm//@angular/core",
        "@org_xprof//frontend/app/common/angular:angular_material_card",
        "@npm//@ngrx/store",
        "@org_xprof//frontend/app/common/constants",
        "@org_xprof//frontend/app/common/interfaces",
        "@org_xprof//frontend/app/common/utils",
        "@org_xprof//frontend/app/store",
    ],
)



sass_binary(
    name = "pod_viewer_details_css",
    src = "pod_viewer_details.scss",

    sourcemap = False,
    deps = [
        "@org_xprof//frontend/app/styles:common",
    ],
)
