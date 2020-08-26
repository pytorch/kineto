load("@npm_angular_bazel//:index.bzl", "ng_module")
load("@io_bazel_rules_sass//:defs.bzl", "sass_binary")

package(default_visibility = ["//frontend:internal"])

ng_module(
    name = "host_side_analysis_detail",
    srcs = [
        "host_side_analysis_detail.ts",
        "host_side_analysis_detail_module.ts",
    ],
    assets = [
        ":host_side_analysis_detail_css",
        "host_side_analysis_detail.ng.html",
    ],
    deps = [
        "@npm//@angular/core",
        "@org_xprof//frontend/app/common/angular:angular_material_expansion",
        "@org_xprof//frontend/third_party/google_visualization",
        "@org_xprof//frontend/app/common/interfaces",
    ],
)



sass_binary(
    name = "host_side_analysis_detail_css",
    src = "host_side_analysis_detail.scss",

    sourcemap = False,
    deps = [
        "@org_xprof//frontend/app/styles:common",
    ],
)
