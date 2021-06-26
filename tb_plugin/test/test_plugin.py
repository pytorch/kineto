import unittest

from tensorboard.plugins.base_plugin import TBContext

from torch_tb_profiler.plugin import TorchProfilerPlugin


class TestPlugin(unittest.TestCase):
    def test_is_active_returns_false_when_no_data(self):
        """Verifies that the plugin correctly reports is_active() == False when no data is available."""

        context = TBContext(logdir="/tmp/nonexistent_logdir/")
        plugin = TorchProfilerPlugin(context)

        self.assertFalse(plugin.is_active())
