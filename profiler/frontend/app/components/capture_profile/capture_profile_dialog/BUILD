load("@npm_angular_bazel//:index.bzl", "ng_module")
load("@io_bazel_rules_sass//:defs.bzl", "sass_binary")

package(default_visibility = ["//frontend:internal"])

ng_module(
    name = "capture_profile_dialog",
    srcs = [
        "capture_profile_dialog.ts",
        "capture_profile_dialog_module.ts",
    ],
    assets = [
        ":capture_profile_dialog_css",
        "capture_profile_dialog.ng.html",
    ],
    deps = [
        "@npm//@angular/common",
        "@npm//@angular/core",
        "@npm//@angular/forms",
        "@org_xprof//frontend/app/common/angular:angular_material_button",
        "@org_xprof//frontend/app/common/angular:angular_material_dialog",
        "@org_xprof//frontend/app/common/angular:angular_material_form_field",
        "@org_xprof//frontend/app/common/angular:angular_material_input",
        "@org_xprof//frontend/app/common/angular:angular_material_radio",
        "@org_xprof//frontend/app/common/angular:angular_material_select",
        "@org_xprof//frontend/app/common/angular:angular_material_tooltip",
        "@npm//@angular/platform-browser",
        "@org_xprof//frontend/app/common/interfaces",
    ],
)



sass_binary(
    name = "capture_profile_dialog_css",
    src = "capture_profile_dialog.scss",

    sourcemap = False,
    deps = [
        "@org_xprof//frontend/app/styles:common",
    ],
)
