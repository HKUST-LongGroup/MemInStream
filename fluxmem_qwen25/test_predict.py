"""CPU contract tests; these do not claim to validate GPU numerical results."""
import ast
import json
import importlib.util
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch

import predict


class PredictionTests(unittest.TestCase):
    def setUp(self):
        self.root = predict.HERE / "upstream"
        self.row = dict(source_dataset="test", source_id="one", question="Which object?",
                        options=["red", "blue", "green", "white"], correct_option="B",
                        answer="blue", query_time=500, video_path="/missing.mp4",
                        evidence=[[2, 4]], evaluation_protocol="backward")

    def test_prompt_uses_filtered_template_and_excludes_annotations(self):
        prompt = predict.make_prompt(self.row, 244, 500)
        self.assertEqual(prompt,
            "The supplied video shows original-video time 244.000s to 500.000s. "
            "The question is asked at 500.000s. "
            "Video timestamps are clip-local; add 244.000s for original-video time.\n"
            "Question: Which object?\nA. red\nB. blue\nC. green\nD. white\n"
            "Select all correct options. Select multiple options only when each is independently "
            "supported by the video and any provided dialogue history. "
            "Do not select additional options merely because you are uncertain. "
            "Your answer is correct only if the selected options exactly match all correct options; "
            "selecting any incorrect option or missing any correct option makes the answer incorrect. "
            "Return only the selected option letter(s), separated by commas, without explanation.")
        changed = dict(self.row, answer="SECRET", correct_option="A", correct_index=0,
                       evidence=[[123, 456]], level1="SECRET", level2="SECRET")
        self.assertEqual(predict.make_prompt(changed, 244, 500), prompt)

    def test_fifth_option_and_multiselect_are_not_lost(self):
        row = dict(self.row, options=["A. a", "B. b", "C. c", "D. d", "E. e"])
        prompt = predict.make_prompt(row, 244, 500)
        self.assertIn("E. e", prompt)
        self.assertNotIn("A. A.", prompt)
        row["correct_option"] = ["A", "C"]
        self.assertEqual(predict.make_prompt(row, 244, 500), prompt)
        row["QA_type"] = "mc_multi"
        multi = predict.make_prompt(row, 244, 500)
        self.assertIn("Do not select additional options merely because you are uncertain.", multi)
        self.assertNotIn("two correct", multi)
        self.assertEqual(prompt, multi)

    def test_binary_open_and_multi_select_prompts(self):
        binary = dict(self.row, options=None, question_type="binary")
        self.assertTrue(predict.make_prompt(binary, 0, 500).endswith("Return only True or False."))
        opened = dict(binary, question_type="open-ended")
        self.assertTrue(predict.make_prompt(opened, 0, 500).endswith(
            "Give a concise, complete answer in the same language as the question. Return only the answer."))
        multi = dict(self.row, question_type="multiple-select", query_time="00:08:20")
        self.assertIn("asked at 500.000s", predict.make_prompt(multi, 244, 500))
        self.assertTrue(predict.make_prompt(multi, 244, 500).endswith(
            "Your answer is correct only if the selected options exactly match all correct options; "
            "selecting any incorrect option or missing any correct option makes the answer incorrect. "
            "Return only the selected option letter(s), separated by commas, without explanation."))

    def contextual_row(self):
        return dict(self.row, dialogue_events=[
            dict(event_id="q1", time=14, role="user", content="What color is the brush?"),
            dict(event_id="a1", time=18, role="assistant", content="Red."),
            dict(event_id="q2", time=26, role="user", content="What is the silver tool?"),
            dict(event_id="a2", time=30, role="assistant", content="A rotary can opener."),
            dict(event_id="a3", time=500, role="assistant", content="SECRET_AT_QUERY"),
            dict(event_id="a4", time=510, role="assistant", content="SECRET_FUTURE"),
        ], required_context_event_ids=["q2", "a2"],
            question="Which compartment holds the utensil from my second question?",
            answer="SECRET_CURRENT_ANSWER")

    def test_contextual_prompt_preserves_prior_turns_and_excludes_current_answer(self):
        row = self.contextual_row()
        prompt = predict.make_prompt(row, 496, 500)
        self.assertIn("a1 [18.000s] assistant: Red.", prompt)
        self.assertIn("a2 [30.000s] assistant: A rotary can opener.", prompt)
        self.assertLess(prompt.index("q1 ["), prompt.index("q2 ["))
        self.assertLess(prompt.index("a2 ["), prompt.index("Question:"))
        self.assertNotIn("SECRET", prompt)
        row.update(answer="CHANGED_GOLD", correct_option="A", correct_index=0)
        self.assertEqual(prompt, predict.make_prompt(row, 496, 500))

    def test_raw_question_fields_work_through_preparation(self):
        row = self.contextual_row()
        for old, new in (("question", "Question"), ("query_time", "Question time"), ("options", "Options")):
            row[new] = row.pop(old)
        args = SimpleNamespace(limit=0, num_shards=1, shard_index=0,
                               path_prefix=[], time_window=4)
        plan, = predict.prepare_rows([row], args)
        self.assertEqual((plan["video_start"], plan["video_end"]), (496, 500))
        self.assertEqual(plan["prompt"], predict.make_prompt(self.contextual_row(), 496, 500))

    def test_matches_reference_function_when_available(self):
        reference = Path(os.environ.get("QWEN_PROMPT_REFERENCE",
            str(predict.HERE.parent / "qwen-eval/predict_filtered_qwen.py")))
        if not reference.is_file():
            self.skipTest("Reference workspace is not included in the standalone package")
        with patch.object(sys, "path", [str(reference.parent), *sys.path]):
            spec = importlib.util.spec_from_file_location("reference_filtered_qwen", reference)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
        rows = [dict(self.row, **changes) for changes in
                       ({}, {"options": ["A) red", "B. blue", "C. green", "D. white", "E. black"]},
                        {"QA_type": "mc_multi"}, {"question_type": "multiple-select"},
                        {"options": [], "question_type": "binary"},
                        {"options": None, "question_type": "open-ended"})]
        rows.append(self.contextual_row())
        rows.append(dict(self.row, dialogue_history=[
            dict(timestamp=30, question="Which color?", options=["Red", "Blue"], answer="B"),
            dict(timestamp=10, role="user", content="Remember this tool."),
        ]))
        raw = self.contextual_row()
        for old, new in (("question", "Question"), ("query_time", "Question time"), ("options", "Options")):
            raw[new] = raw.pop(old)
        rows.append(raw)
        for row in rows:
            for start in (0, 244, 123.456789):
                self.assertEqual(predict.make_prompt(row, start, 500), module.make_prompt(row, start, 500))

    def test_modified_upstream_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            import shutil
            root = Path(directory) / "upstream"
            shutil.copytree(self.root, root)
            predict.verify_upstream(root)
            target = root / "models/qwen2-5-vl/src/qwen2_5_vl_fluxmem/fluxmem.py"
            target.write_text(target.read_text() + "\n# experimental modification\n")
            with self.assertRaisesRegex(ValueError, "verification failed"):
                predict.verify_upstream(root)

    def test_shards_cover_input_and_path_mapping_respects_boundaries(self):
        rows = [dict(self.row, source_id=str(i)) for i in range(7)]
        all_indices = []
        for rank in range(3):
            args = SimpleNamespace(limit=0, num_shards=3, shard_index=rank,
                                   path_prefix=[], time_window=256)
            plans = predict.prepare_rows(rows, args)
            all_indices.extend(p["input_index"] for p in plans)
            self.assertTrue(all(p["video_start"] == 244 for p in plans))
        self.assertEqual(sorted(all_indices), list(range(7)))
        self.assertEqual(predict.remap_path("/old/video.mp4", [("/old", "/new")]), "/new/video.mp4")
        self.assertEqual(predict.remap_path("/older/video.mp4", [("/old", "/new")]), "/older/video.mp4")

    def test_local_ovo_inference_contract_and_chunk_forwarding(self):
        # Execute the exact copied get_response function with external GPU objects mocked.
        tree = ast.parse((self.root / predict.EVALUATOR).read_text())
        func = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "get_response")
        process = MagicMock(return_value=(None, ["video_tensor"], {"fps": [0.75]}))
        env = dict(torch=MagicMock(), process_vision_info=process,
                   MIN_PIXELS=12544, MAX_PIXELS=401408, MIN_FRAMES=4, MAX_FRAMES=256, FPS=1)
        exec(compile(ast.Module(body=[func], type_ignores=[]), "local_ovo", "exec"), env)
        evaluator = predict.LocalFluxMemPredictor.__new__(predict.LocalFluxMemPredictor)
        evaluator.module = SimpleNamespace(get_response=env["get_response"])
        evaluator.args = SimpleNamespace(short_frames=8, medium_frames=64, video_chunk_t=4)
        evaluator.stats_path = Path("stats.jsonl")
        evaluator.processor = MagicMock()
        evaluator.processor.apply_chat_template.return_value = "chat"
        inputs = MagicMock()
        inputs.input_ids = [[0] * 12]
        evaluator.processor.return_value.to.return_value = inputs
        evaluator.model = MagicMock()
        evaluator.model.generate.return_value = [[0] * 12 + [99]]
        evaluator.processor.batch_decode.return_value = ["E"]
        plan = dict(video_path="clip.mp4", prompt="prompt", video_start=244.0,
                    video_end=500, anchor_end=True, input_index=10)
        self.assertEqual(evaluator.predict(plan), "E")
        video, text = process.call_args.args[0][0]["content"]
        self.assertEqual(video, dict(type="video", video="clip.mp4", min_pixels=12544,
            max_pixels=401408, max_frames=256, min_frames=4, fps=1,
            video_start=244.0, video_end=500, anchor_end=True, sample_id=10))
        self.assertEqual(text, dict(type="text", text="prompt"))
        self.assertEqual(evaluator.model.generate.call_args.kwargs,
            dict(max_new_tokens=128, use_fluxmem=True, short_frames=8, medium_frames=64,
                 save_path="stats.jsonl", video_chunk_t=4))
        self.assertEqual(evaluator.processor.call_args.kwargs["fps"], [0.75])
        evaluator.processor.batch_decode.assert_called_once_with(
            [[99]], skip_special_tokens=True, clean_up_tokenization_spaces=False)
        evaluator.args.video_chunk_t = None
        plan.update(video_start=0.0, anchor_end=False)
        evaluator.predict(plan)
        self.assertNotIn("video_chunk_t", evaluator.model.generate.call_args.kwargs)
        video = process.call_args.args[0][0]["content"][0]
        self.assertEqual((video["video_start"], video["video_end"], video["anchor_end"]), (0, 500, False))

    def test_bundled_source_manifest_has_no_local_source_dependency(self):
        manifest = predict.verify_upstream(self.root)
        self.assertNotIn('source_path', manifest)
        self.assertEqual(manifest['source_kind'], 'bundled_modified_snapshot')
        self.assertTrue(manifest['contains_local_changes'])

    def test_resume_retries_errors_without_duplicating_successes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            video = root / "video.mp4"
            video.touch()
            rows = [dict(self.row, video_path=str(video), source_id=str(i)) for i in range(2)]
            source = root / "data.json"
            source.write_text(json.dumps(rows))
            output = root / "predictions.jsonl"
            argv = ["--input", str(source), "--output", str(output)]
            evaluator = MagicMock()
            evaluator.diagnostics = {}
            evaluator.predict.side_effect = ["B", RuntimeError("decode failed")]
            with patch.object(predict, "environment_versions", return_value={}), patch.object(
                    predict, "load_official", return_value=evaluator):
                self.assertEqual(predict.main(argv), 1)
                evaluator.predict.side_effect = ["C"]
                self.assertEqual(predict.main(argv), 0)
                self.assertEqual(evaluator.predict.call_count, 3)
                self.assertEqual(predict.main(argv), 0)
                self.assertEqual(evaluator.predict.call_count, 3)
                with self.assertRaisesRegex(ValueError, "settings/input/source/environment changed"):
                    predict.main(argv + ["--time-window", "0"])
            records = [json.loads(line) for line in output.read_text().splitlines()]
            self.assertEqual(len(records), 2)
            self.assertEqual({r["source_id"] for r in records}, {"0", "1"})

    def test_gpu_selection_respects_visible_ids(self):
        with patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "2,5,7"}):
            self.assertEqual(predict.select_gpus(2, dry_run=True), ["2", "5"])
            with self.assertRaises(ValueError):
                predict.select_gpus(4, dry_run=True)

    def test_consecutive_failure_limit_preserves_traceback_and_unattempted_rows(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            video = root / "video.mp4"
            video.touch()
            source = root / "data.json"
            source.write_text(json.dumps([dict(self.row, source_id=str(i), video_path=str(video)) for i in range(5)]))
            output = root / "predictions.jsonl"
            evaluator = MagicMock()
            evaluator.diagnostics = {"visual_calls": [{"grid_thw": [[128, 32, 64]]}]}
            evaluator.predict.side_effect = RuntimeError("synthetic decode failure")
            with patch.object(predict, "environment_versions", return_value={}), patch.object(
                    predict, "load_official", return_value=evaluator):
                code = predict.main(["--input", str(source), "--output", str(output),
                                     "--max-consecutive-errors", "2"])
            self.assertEqual(code, 1)
            self.assertEqual(evaluator.predict.call_count, 2)
            self.assertEqual(evaluator.release_after_error.call_count, 1)
            summary = json.loads(Path(str(output) + ".summary.json").read_text())
            self.assertEqual((summary["completed"], summary["remaining"], summary["failed_this_run"]), (0, 5, 2))
            errors = [json.loads(line) for line in Path(str(output) + ".errors.jsonl").read_text().splitlines()]
            self.assertIn("Traceback", errors[0]["traceback"])
            self.assertEqual(errors[0]["diagnostics"], evaluator.diagnostics)

    def test_loader_matches_local_ovo_settings_and_selects_requested_attention(self):
        model = SimpleNamespace(config=SimpleNamespace(model_type="qwen2_5_vl"),
            visual=SimpleNamespace(blocks=[SimpleNamespace(attn=object())], register_forward_pre_hook=MagicMock()),
            model=SimpleNamespace(layers=[SimpleNamespace(self_attn=object())]),
            generation_config=SimpleNamespace(to_dict=lambda: {"do_sample": True}))
        module = SimpleNamespace(torch=MagicMock(),
            Qwen2_5_VLForConditionalGeneration=MagicMock(), Qwen2_5_VLProcessor=MagicMock())
        module.Qwen2_5_VLForConditionalGeneration.from_pretrained.return_value = model
        args = SimpleNamespace(model_path="checkpoint", min_pixels=12544, max_pixels=401408,
            min_frames=4, max_frames=256, fps=1, attn_implementation="sdpa", video_chunk_t=4)
        instance = predict.LocalFluxMemPredictor(module, args, Path("stats.jsonl"))
        module.Qwen2_5_VLForConditionalGeneration.from_pretrained.assert_called_once_with(
            "checkpoint", torch_dtype=module.torch.bfloat16, attn_implementation="sdpa", device_map="auto")
        module.torch.manual_seed.assert_called_once_with(1234)
        self.assertEqual(instance.processor.tokenizer.model_max_length, 32768)
        self.assertEqual(module.MIN_FRAMES, 4)
        self.assertEqual(model.generation_config.to_dict(), {"do_sample": True})

    def test_sdpa_allows_absent_flash_attn(self):
        versions = {"torch": "2.6.0", "torchvision": "0.21.0", "transformers": "4.50.0", "accelerate": "1.4.0"}
        def version(name):
            if name not in versions:
                raise predict.importlib.metadata.PackageNotFoundError(name)
            return versions[name]
        with patch.object(predict.importlib.metadata, "version", side_effect=version):
            self.assertIsNone(predict.environment_versions("sdpa")["flash-attn"])
            with self.assertRaisesRegex(RuntimeError, "FlashAttention 2"):
                predict.environment_versions("flash_attention_2")

    def test_existing_torch_environment_is_not_rejected_by_version_pin(self):
        versions = {"torch": "2.5.1+cu121", "torchvision": "0.20.1+cu121",
                    "transformers": "4.57.1", "accelerate": "1.2.0"}
        def version(name):
            if name not in versions:
                raise predict.importlib.metadata.PackageNotFoundError(name)
            return versions[name]
        with patch.object(predict.importlib.metadata, "version", side_effect=version):
            with patch("sys.stderr") as stderr:
                result = predict.environment_versions("sdpa")
            self.assertEqual(result["torch"], "2.5.1+cu121")
            self.assertTrue(stderr.write.called)
            del versions["transformers"]
            with patch("sys.stderr"), self.assertRaisesRegex(RuntimeError, "Required package transformers"):
                predict.environment_versions("sdpa")

    def test_worker_command_forwards_configuration_without_recursive_launch(self):
        args = SimpleNamespace(input=Path("input.json"), fluxmem_root=self.root,
            model_path="YOUR_MODEL/qwen", num_gpus=4, limit=10, time_window=600,
            attn_implementation="sdpa",
            fps=0.5, min_frames=4, max_frames=256, min_pixels=12544, max_pixels=401408, short_frames=8,
            video_chunk_t=4, cpu_threads=1, max_consecutive_errors=5,
            medium_frames=64, path_prefix=[("/old", "/new")])
        command = predict.worker_command(args, 2, Path("output.jsonl"))
        self.assertEqual(command[command.index("--video-chunk-t")+1], "4")
        self.assertEqual(command[command.index("--min-frames")+1], "4")
        self.assertEqual(command[command.index("--cpu-threads")+1], "1")
        self.assertNotIn("--num-gpus", command)
        self.assertEqual(command[command.index("--attn-implementation")+1], "sdpa")
        self.assertEqual(command[command.index("--num-shards")+1], "4")
        self.assertEqual(command[command.index("--shard-index")+1], "2")
        self.assertEqual(command[command.index("--time-window")+1], "600")
        self.assertEqual(command[command.index("--path-prefix")+1], "/old=/new")

    def test_launcher_real_subprocesses_merge_and_retry_failure(self):
        # Real CPU child processes exercise dispatch, GPU masking, exit handling and merge.
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "merged.jsonl"
            predict.write_json(Path(str(output) + ".manifest.json"), {"test": "launcher"})
            plans = [dict(input_index=i, source_dataset="test", source_id=str(i)) for i in range(5)]
            args = SimpleNamespace(num_gpus=2)
            fail_rank = [1]
            def command(_args, rank, path):
                program = (
                    "import json,os,sys; from pathlib import Path\n"
                    f"rank={rank}; path=Path({str(path)!r})\n"
                    f"if rank=={fail_rank[0]}: sys.exit(2)\n"
                    "done={json.loads(s)['fluxmem_prediction']['input_index'] for s in path.read_text().splitlines()} if path.exists() else set()\n"
                    "with path.open('a') as f:\n"
                    " for i in range(5):\n"
                    "  if i%2==rank and i not in done:\n"
                    "   f.write(json.dumps(dict(source_dataset='test',source_id=str(i),fluxmem_prediction=dict(input_index=i,status='completed',response=os.environ['CUDA_VISIBLE_DEVICES'])))+'\\n')\n")
                return [sys.executable, "-c", program]
            with patch.object(predict, "worker_command", side_effect=command):
                self.assertEqual(predict.run_multi_gpu(args, output, plans, {}, ["2", "5"]), 1)
                summary = json.loads(Path(str(output) + ".summary.json").read_text())
                self.assertEqual((summary["completed"], summary["remaining"]), (3, 2))
                fail_rank[0] = -1
                self.assertEqual(predict.run_multi_gpu(args, output, plans, {}, ["2", "5"]), 0)
                records = [json.loads(s) for s in output.read_text().splitlines()]
                self.assertEqual([r["fluxmem_prediction"]["input_index"] for r in records], list(range(5)))
                self.assertEqual([r["fluxmem_prediction"]["response"] for r in records], ["2", "5", "2", "5", "2"])
                self.assertEqual(predict.run_multi_gpu(args, output, plans, {}, ["2", "5"]), 0)
                self.assertEqual(len(output.read_text().splitlines()), 5)


if __name__ == "__main__":
    unittest.main()
