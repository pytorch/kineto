load("@npm_angular_bazel//:index.bzl", "ng_module")
load("@io_bazel_rules_sass//:defs.bzl", "sass_binary", "sass_library")

package(default_visibility = ["//frontend:internal"])

ng_module(
    name = "input_pipeline",
    srcs = [
        "input_pipeline.ts",
        "input_pipeline_common.ts",
        "input_pipeline_module.ts",
    ],
    assets = [
        ":input_pipeline_css",
        "input_pipeline.ng.html",
    ],
    deps = [
        "@npm//@angular/core",
        "@org_xprof//frontend/app/common/angular:angular_material_divider",
        "@npm//@angular/router",
        "@npm//@ngrx/store",
        "@org_xprof//frontend/app/common/interfaces",
        "@org_xprof//frontend/app/common/utils",
        "@org_xprof//frontend/app/components/diagnostics_view",
        "@org_xprof//frontend/app/components/input_pipeline/analysis_summary",
        "@org_xprof//frontend/app/components/input_pipeline/device_side_analysis_detail",
        "@org_xprof//frontend/app/components/input_pipeline/host_side_analysis_detail",
        "@org_xprof//frontend/app/services/data_service",
    ],
)



sass_binary(
    name = "input_pipeline_css",
    src = "input_pipeline.scss",

    sourcemap = False,
    deps = [
        "@org_xprof//frontend/app/styles:common",
    ],
)

sass_library(
    name = "common_css",
    srcs = ["input_pipeline.scss"],
    deps = [
        "@org_xprof//frontend/app/styles:common",
    ],
)
