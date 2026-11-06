"""CPU regressions for PD rank selection and padded FA3 draft scales.

Execute the affected source methods without importing CUDA/communication
backends. This keeps these host-side checks runnable without serving hardware.
"""

import ast
from pathlib import Path
from types import SimpleNamespace
import time
import unittest
from unittest.mock import Mock

import torch

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


ROOT = Path(__file__).resolve().parents[3] / "python/sglang/srt"


def source_class(path, name):
    return next(
        node
        for node in ast.parse((ROOT / path).read_text()).body
        if isinstance(node, ast.ClassDef) and node.name == name
    )


def execute(nodes, namespace):
    module = ast.Module(
        body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)] + nodes,
        type_ignores=[],
    )
    exec(compile(ast.fix_missing_locations(module), "<production methods>", "exec"), namespace)


class TestPrefillRank(unittest.TestCase):
    def sender(self, *, system_size, system_rank, attention_rank, room,
               policy="follow_bootstrap_room", injected=False, force=False):
        cls = source_class("disaggregation/common/conn.py", "CommonKVSender")
        methods = [n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name in
                   {"__init__", "_prefill_dp_rank", "_register_prefill_dp_rank"}]
        namespace = {
            "KVTransferMetric": Mock, "TraceNullContext": Mock,
            "KVPoll": SimpleNamespace(Bootstrapping="bootstrap", Failed="failed", WaitingForInput="waiting"),
            "get_parallel": lambda: SimpleNamespace(num_dp_ranks=4, load_balance_method=policy),
            "envs": SimpleNamespace(SGLANG_DISAGGREGATION_FORCE_QUERY_PREFILL_DP_RANK=SimpleNamespace(get=lambda: force)),
            "requests": SimpleNamespace(post=Mock(return_value=SimpleNamespace(status_code=200))),
            "logger": Mock(), "time": time,
        }
        execute([ast.ClassDef(name="Sender", bases=[], keywords=[], body=methods, decorator_list=[])], namespace)
        mgr = SimpleNamespace(
            system_dp_size=system_size, system_dp_rank=system_rank, attn_dp_rank=attention_rank,
            sender_trace_module=False, deferred_bootstrap=None, is_dummy_cp_rank=False,
            update_status=Mock(), record_failure=Mock(),
        )
        sender = namespace["Sender"](mgr, "localhost:8998", room, injected)
        return sender, mgr, namespace["requests"].post

    def test_matching_bootstrap_room_accepts_both_dp_modes(self):
        for system_size, system_rank, attention_rank in [(4, 2, 0), (1, 0, 2)]:
            with self.subTest(system_size=system_size):
                _, mgr, post = self.sender(system_size=system_size, system_rank=system_rank,
                                           attention_rank=attention_rank, room=2)
                mgr.record_failure.assert_not_called()
                post.assert_not_called()

    def test_conflicting_room_is_rejected_with_the_real_rank(self):
        for system_size, system_rank, attention_rank in [(4, 2, 0), (1, 0, 2)]:
            with self.subTest(system_size=system_size):
                _, mgr, _ = self.sender(system_size=system_size, system_rank=system_rank,
                                        attention_rank=attention_rank, room=0)
                mgr.record_failure.assert_called_once()
                self.assertIn("dispatched to dp_rank 2", mgr.record_failure.call_args.args[1])
                self.assertEqual(mgr.update_status.call_args.args, (0, "failed"))

    def test_registration_uses_the_real_rank(self):
        for topology in [(4, 2, 0), (1, 0, 2)]:
            for policy, force in [("round_robin", False), ("follow_bootstrap_room", True)]:
                with self.subTest(topology=topology, policy=policy):
                    _, mgr, post = self.sender(system_size=topology[0], system_rank=topology[1],
                                               attention_rank=topology[2], room=0, policy=policy, force=force)
                    self.assertEqual(post.call_args.kwargs["json"]["dp_rank"], 2)
                    mgr.record_failure.assert_not_called()

    def test_router_selected_rank_bypasses_registration(self):
        _, mgr, post = self.sender(system_size=4, system_rank=2, attention_rank=0,
                                   room=0, policy="round_robin", injected=True)
        mgr.record_failure.assert_not_called()
        post.assert_not_called()


class TestDraftScaleRows(unittest.TestCase):
    def run_prepare(self, *, sequences=5, batch_size=3, draft=True, version=3,
                    mla=False, local=False, swa=False, cross=False, cp=False):
        cls = source_class("layers/attention/flashattention_backend.py", "FlashAttentionBackend")
        methods = {n.name: n for n in cls.body if isinstance(n, ast.FunctionDef)}
        body = methods["forward_extend"].body
        start = next(i for i, n in enumerate(body) if isinstance(n, ast.Assign)
                     and ast.unparse(n.targets[0]) == "metadata")
        end = next(i for i, n in enumerate(body) if isinstance(n, ast.Assign)
                   and isinstance(n.value, ast.Call) and isinstance(n.value.func, ast.Attribute)
                   and n.value.func.attr == "prepare_paged_mha_query")
        namespace = {"torch": torch}
        execute([methods["prepare_paged_mha_query"]], namespace)
        backend = SimpleNamespace(
            forward_metadata=SimpleNamespace(cu_seqlens_q=torch.arange(sequences + 1, dtype=torch.int32)),
            fa_impl_ver=version, use_mla=mla, kv_cache_dtype_str="fp8_e4m3",
            kv_cache_dtype=torch.float8_e4m3fn, kv_cache_is_mxfp8=False,
            local_attn_builder=SimpleNamespace(applies=lambda *args: True) if local else None,
        )
        backend.prepare_paged_mha_query = lambda *args, **kwargs: namespace["prepare_paged_mha_query"](backend, *args, **kwargs)
        layer = SimpleNamespace(sliding_window_size=32 if swa else None,
                                is_cross_attention=cross, attn_type="causal", head_dim=16,
                                tp_k_head_num=2, k_scale=torch.tensor(2.0), v_scale=torch.tensor(3.0))
        namespace.update(self=backend, layer=layer, cp_active=cp,
                         forward_batch=SimpleNamespace(batch_size=batch_size, forward_mode=SimpleNamespace(is_draft_extend_v2=lambda: draft)),
                         AttentionType=SimpleNamespace(ENCODER_ONLY="encoder", DECODER_BIDIRECTIONAL="bidi"),
                         q=torch.ones(7, 2, 16), q_rope=None, k_rope=None)
        execute(body[start:end + 1], namespace)
        return layer, namespace

    def test_padded_draft_uses_query_sequence_count_without_changing_scales(self):
        for count in (1, 3, 5):
            with self.subTest(sequences=count):
                layer, ns = self.run_prepare(sequences=count)
                for name, scale in [("fa_k_descale", layer.k_scale), ("fa_v_descale", layer.v_scale)]:
                    result = ns[name]
                    self.assertEqual(result.shape, (count, 2))
                    self.assertTrue(torch.equal(result, torch.full((count, 2), scale.item())))
                    self.assertEqual(result.data_ptr(), scale.data_ptr())

    def test_other_attention_paths_keep_their_existing_batch_size(self):
        for kwargs in [dict(draft=False), dict(mla=True), dict(local=True),
                       dict(swa=True), dict(cross=True), dict(cp=True)]:
            with self.subTest(**kwargs):
                _, ns = self.run_prepare(**kwargs)
                self.assertEqual(ns["fa_k_descale"].shape, (3, 2))

    def test_fa4_keeps_its_existing_prefill_scale_behavior(self):
        _, ns = self.run_prepare(version=4)
        self.assertIsNone(ns["fa_k_descale"])
        self.assertIsNone(ns["fa_v_descale"])


if __name__ == "__main__":
    unittest.main()
