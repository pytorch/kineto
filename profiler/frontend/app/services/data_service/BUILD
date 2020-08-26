load("@npm_angular_bazel//:index.bzl", "ng_module")

package(default_visibility = ["//frontend:internal"])

ng_module(
    name = "data_service",
    srcs = [
        "data_service.ts",
        "mock_data.ts",
    ],
    deps = [
        "@npm//@angular/common",
        "@org_xprof//frontend/app/common/angular:angular_common_http",
        "@npm//@angular/core",
        "@npm//rxjs",
        "@org_xprof//frontend/app/common/constants",
        "@org_xprof//frontend/app/common/interfaces",
    ],
)


