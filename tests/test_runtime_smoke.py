import importlib.util
import os
import sys
import unittest
from pathlib import Path

try:
    import torch
except ImportError:  # pragma: no cover - optional ComfyUI environment
    torch = None

COMFYUI_ROOT = Path(os.environ.get("COMFYUI_PATH", r"F:\SDComfyUI"))
PLUGIN_ROOT = Path(__file__).resolve().parents[1]


@unittest.skipUnless(torch is not None and (COMFYUI_ROOT / "comfy").is_dir(), "requires the local ComfyUI Python environment")
class ComfyRuntimeSmokeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        sys.path.insert(0, str(COMFYUI_ROOT))
        sys.path.insert(0, str(PLUGIN_ROOT))
        module_path = PLUGIN_ROOT / "nodes_dynamic_lora_rank.py"
        spec = importlib.util.spec_from_file_location("dynamic_lora_rank_runtime", module_path)
        cls.module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(cls.module)
        from comfy.weight_adapter.lora import LoRAAdapter
        cls.LoRAAdapter = LoRAAdapter

    def test_transformer_linear_rank_mask_uses_last_axis(self):
        m = self.module
        up = torch.ones(2, 4)
        down = torch.ones(4, 3)
        adapter = m.ScheduledLoRAAdapter(
            self.LoRAAdapter(set(), (up, down, 4.0, None, None, None)),
            (0.5,),
            (1.0,),
        )
        previous = m._set_current_step(0)
        output = adapter.h(torch.ones(2, 5, 3), torch.zeros(2, 5, 2))
        m._restore_current_step(previous)
        self.assertEqual(tuple(output.shape), (2, 5, 2))
        self.assertTrue(torch.allclose(output, torch.full_like(output, 6.0)))

    def test_conv_rank_mask_uses_channel_axis(self):
        m = self.module
        up = torch.ones(2, 4, 1, 1)
        down = torch.ones(4, 3, 1, 1)
        adapter = m.ScheduledLoRAAdapter(
            self.LoRAAdapter(set(), (up, down, 4.0, None, None, None)),
            (0.5,),
            (1.0,),
        )
        adapter.is_conv = True
        adapter.conv_dim = 2
        adapter.kernel_size = (1, 1)
        adapter.in_channels = 3
        adapter.kw_dict = {"stride": (1, 1), "padding": (0, 0), "dilation": (1, 1), "groups": 1}
        previous = m._set_current_step(0)
        output = adapter.h(torch.ones(2, 3, 4, 4), torch.zeros(2, 2, 4, 4))
        m._restore_current_step(previous)
        self.assertEqual(tuple(output.shape), (2, 2, 4, 4))

    def test_serial_adapters_keep_independent_schedules(self):
        m = self.module
        up = torch.ones(2, 4)
        down = torch.ones(4, 3)
        adapter_a = m.ScheduledLoRAAdapter(
            self.LoRAAdapter(set(), (up, down, 4.0, None, None, None)),
            (1.0, 0.0),
            (1.0, 0.0),
        )
        adapter_b = m.ScheduledLoRAAdapter(
            self.LoRAAdapter(set(), (up, down, 4.0, None, None, None)),
            (0.0, 1.0),
            (0.0, 1.0),
        )
        previous = m._set_current_step(1)
        output_a = adapter_a.h(torch.ones(2, 5, 3), torch.zeros(2, 5, 2))
        output_b = adapter_b.h(torch.ones(2, 5, 3), torch.zeros(2, 5, 2))
        m._restore_current_step(previous)
        self.assertEqual(float(output_a.abs().sum()), 0.0)
        self.assertGreater(float(output_b.abs().sum()), 0.0)

    def test_strength_schedule_scales_complete_output(self):
        m = self.module
        up = torch.ones(2, 4)
        down = torch.ones(4, 3)
        adapter = m.ScheduledLoRAAdapter(
            self.LoRAAdapter(set(), (up, down, 4.0, None, None, None)),
            (1.0,),
            (0.5,),
        )
        previous = m._set_current_step(0)
        output = adapter.h(torch.ones(2, 5, 3), torch.zeros(2, 5, 2))
        m._restore_current_step(previous)
        self.assertTrue(torch.allclose(output, torch.full_like(output, 6.0)))
