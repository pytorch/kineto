load("@npm_angular_bazel//:index.bzl", "ng_module")
load("@io_bazel_rules_sass//:defs.bzl", "sass_binary")

package(default_visibility = ["//frontend:internal"])

ng_module(
    name = "app",
    srcs = [
        "app.ts",
        "app_module.ts",
    ],
    assets = [
        ":app_css",
        "app.ng.html",
    ],
    deps = [
        "@org_xprof//frontend/app/common/angular:angular_common_http",
        "@npm//@angular/core",
        "@org_xprof//frontend/app/common/angular:angular_material_progress_bar",
        "@npm//@angular/platform-browser",
        "@org_xprof//frontend/app/common/angular:angular_platform-browser_animation",
        "@org_xprof//frontend/app/common/classes",
        "@org_xprof//frontend/app/common/constants",
        "@org_xprof//frontend/app/common/interfaces",
        "@org_xprof//frontend/app/components/empty_page",
        "@org_xprof//frontend/app/components/main_page",
        "@org_xprof//frontend/app/pipes",
        "@org_xprof//frontend/app/services/data_service",
        "@org_xprof//frontend/app/store",
    ],
)



sass_binary(
    name = "app_css",
    src = "app.scss",

    sourcemap = False,
)
