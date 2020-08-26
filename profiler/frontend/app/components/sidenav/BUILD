load("@npm_angular_bazel//:index.bzl", "ng_module")
load("@io_bazel_rules_sass//:defs.bzl", "sass_binary")

package(default_visibility = ["//frontend:internal"])

ng_module(
    name = "sidenav",
    srcs = [
        "sidenav.ts",
        "sidenav_module.ts",
    ],
    assets = [
        ":sidenav_css",
        "sidenav.ng.html",
    ],
    deps = [
        "@npm//@angular/core",
        "@org_xprof//frontend/app/common/angular:angular_material_core",
        "@org_xprof//frontend/app/common/angular:angular_material_form_field",
        "@org_xprof//frontend/app/common/angular:angular_material_select",
        "@npm//@ngrx/store",
        "@org_xprof//frontend/app/common/constants",
        "@org_xprof//frontend/app/common/interfaces",
        "@org_xprof//frontend/app/components/capture_profile",
        "@org_xprof//frontend/app/components/memory_viewer/buffer_details",
        "@org_xprof//frontend/app/components/op_profile/op_details",
        "@org_xprof//frontend/app/components/pod_viewer/pod_viewer_details",
        "@org_xprof//frontend/app/services/data_service",
        "@org_xprof//frontend/app/store",
    ],
)



sass_binary(
    name = "sidenav_css",
    src = "sidenav.scss",

    sourcemap = False,
    deps = [
        "@org_xprof//frontend/app/styles:common",
    ],
)
