
## Build
```
$ bazel run plugin:build_pip_package
```

## Test
```
$ bazel test plugin/...
```

## Using plugin
Inside `kineto/` folder:
```
$ mkdir profile_env
$ python3 profiler/install_and_run.py --envdir=profile_env --logdir=profiler/demo
```
