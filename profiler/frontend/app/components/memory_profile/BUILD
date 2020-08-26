load("@npm_angular_bazel//:index.bzl", "ng_module")
load("@npm_bazel_typescript//:index.bzl", "ts_library")
load("@io_bazel_rules_sass//:defs.bzl", "sass_binary", "sass_library")

package(default_visibility = ["//frontend:internal"])

ng_module(
    name = "memory_profile",
    srcs = [
        "memory_profile.ts",
        "memory_profile_module.ts",
    ],
    assets = [
        ":memory_profile_css",
        "memory_profile.ng.html",
    ],
    deps = [
        ":memory_profile_base",
        "@npm//@angular/core",
        "@org_xprof//frontend/app/common/angular:angular_material_core",
        "@org_xprof//frontend/app/common/angular:angular_material_divider",
        "@org_xprof//frontend/app/common/angular:angular_material_icon",
        "@org_xprof//frontend/app/common/angular:angular_material_select",
        "@npm//@angular/router",
        "@npm//@ngrx/store",
        "@org_xprof//frontend/app/common/interfaces",
        "@org_xprof//frontend/app/components/memory_profile/memory_breakdown_table",
        "@org_xprof//frontend/app/components/memory_profile/memory_profile_summary",
        "@org_xprof//frontend/app/components/memory_profile/memory_timeline_graph",
        "@org_xprof//frontend/app/services/data_service",
        "@org_xprof//frontend/app/store",
    ],
)

ts_library(
    name = "memory_profile_base",
    srcs = [
        "memory_profile_base.ts",
    ],
    deps = [
        "@org_xprof//frontend/app/common/interfaces",
    ],
)



sass_binary(
    name = "memory_profile_css",
    src = "memory_profile.scss",

    sourcemap = False,
    deps = [
        ":memory_profile_common",
    ],
)

sass_library(
    name = "memory_profile_common",
    srcs = ["memory_profile_common.scss"],
    deps = [
        "@org_xprof//frontend/app/styles:common",
    ],
)
