# pytorch_profiler

### Quick Installation Instructions

* Clone the git repository

  `git clone https://github.com/pytorch/kineto.git`

* Navigate to the plugin directory

* Install pytorch_profiler

  `pip install .`

* Verify installation is complete

  `pip list | grep tensorboard-plugin-torch-profiler`

  Should display

  `tensorboard-plugin-torch-profiler 0.1.0`


### Quick Start Instructions

* Start tensorboard

  Specify your profiling samples folder.
  Or you can specify <pytorch_profiler>/samples as an example.

  `tensorboard --logdir=./samples`

  If your web browser is not in the same machine that you start tensorboard,
  you can add `--bind_all` option, such as:

  `tensorboard --logdir=./samples --bind_all`

  Note: Make sure the default port 6006 is open to the browser's host.

* Open tensorboard in Chrome browser

  Open URL `http://localhost:6006` in the browser.

* Navigate to TORCH_PROFILER tab

  If the files under `--logdir` are too big or too many.
  Refresh the browser to check latest loaded result.
