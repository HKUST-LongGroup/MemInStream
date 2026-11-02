import ast
import copy
import importlib.util
import json
import os
from contextlib import nullcontext
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace, ModuleType
import unittest
from unittest.mock import patch

import core
import predict
import evaluate


class Tests(unittest.TestCase):
    def args(self, *extra):
        return predict.parse_args(['--input','input.json','--output','out.jsonl','--model-path','/weights',*extra])

    def test_snapshot(self):
        self.assertEqual(core.verify_upstream()['commit'], '5af0abe438155d33aba69ca59f25b8b3ea9b81d7')

    def test_no_labels_in_plan(self):
        row = dict(video_path='a.mp4',question='Q',query_time=1.5,options=['A. a','B. b'],answer='SECRET',evidence='SECRET')
        plan = predict.prepare(row,self.args())
        self.assertNotIn('SECRET',json.dumps(plan))
        self.assertTrue(plan['prompt_adapted'])
        with self.assertRaises(ValueError): predict.prepare(row,self.args('--profile','streamingbench'))
        row['evaluation_protocol']='proactive'
        with self.assertRaises(ValueError): predict.prepare(row,self.args())

    def test_final_prompt_for_every_question_type(self):
        row = dict(video_path='a', question='Q', query_time=2, question_type='binary')
        self.assertTrue(predict.prepare(row, self.args())['prompt_adapted'])
        choice = dict(row, options=['A', 'B', 'C', 'D'])
        self.assertIn('Select all correct', predict.prepare(choice, self.args())['prompt'])
        choice['dialogue_events'] = [dict(time=1, role='user', content='Remember this.')]
        self.assertIn('Remember this.', predict.prepare(choice, self.args())['prompt'])

    def contextual_row(self):
        return dict(video_path='missing.mp4', query_time=332,
                    question='Where is the utensil from my second question?',
                    options=['A. Left', 'B. Center'], answer='SECRET_GOLD',
                    required_context_event_ids=['q2', 'a2'], dialogue_events=[
            dict(event_id='q1', time=14, role='user', content='Which color?'),
            dict(event_id='a1', time=18, role='assistant', content='Red.'),
            dict(event_id='q2', time=26, role='user', content='Which tool?'),
            dict(event_id='a2', time=30, role='assistant', content='A rotary can opener.'),
            dict(event_id='a3', time=332, role='assistant', content='SECRET_AT_QUERY'),
            dict(event_id='a4', time=340, role='assistant', content='SECRET_FUTURE'),
        ])

    def test_contextual_history_and_scene_graph(self):
        row = self.contextual_row()
        plan = predict.prepare(row, self.args())
        prompt = predict.format_prompt(plan, None, {})
        self.assertIn('a1 [18.000s] assistant: Red.', prompt)
        self.assertIn('a2 [30.000s] assistant: A rotary can opener.', prompt)
        self.assertLess(prompt.index('q1 ['), prompt.index('q2 ['))
        self.assertLess(prompt.index('a2 ['), prompt.index('Question:'))
        self.assertNotIn('SECRET', json.dumps(plan))
        ns = {'format_scene_graphs_for_prompt_offline': lambda graphs: 'SCENE_GRAPH'}
        graph_prompt = predict.format_prompt(plan, [(100, {})], ns)
        self.assertTrue(graph_prompt.endswith(prompt))
        self.assertIn('<100.0 seconds>:SCENE_GRAPH', graph_prompt)
        row.update(answer='CHANGED_GOLD', correct_option='A')
        self.assertEqual(prompt, predict.prepare(row, self.args())['prompt'])

    def test_qwen_prompt_parity(self):
        reference = Path(os.environ.get('QWEN_PROMPT_REFERENCE',
            str(core.HERE.parent / 'qwen-eval/predict_filtered_qwen.py')))
        if not reference.is_file():
            self.skipTest('Qwen reference is not available in standalone package')
        spec = importlib.util.spec_from_file_location('reference_qwen', reference)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        contextual = self.contextual_row()
        raw = copy.deepcopy(contextual)
        for old, new in (('question', 'Question'), ('query_time', 'Question time'), ('options', 'Options')):
            raw[new] = raw.pop(old)
        plain = dict(video_path='missing', query_time=332, question='Which?', options=['Red', 'Blue'])
        rows = [plain, contextual, raw, dict(plain, QA_type='mc_multi'),
                dict(plain, options=None, question_type='binary'),
                dict(plain, options=None, question_type='open-ended'),
                dict(plain, dialogue_history=[dict(timestamp=20, question='Which color?',
                     options=['Red', 'Blue'], answer='B')])]
        for profile in ('ovobench', 'streamingbench'):
            for row in rows:
                plan = predict.prepare(row, self.args('--profile', profile))
                self.assertTrue(plan['prompt_adapted'])
                self.assertEqual(predict.format_prompt(plan, None, {}), module.make_prompt(row, 0, 332))

    def test_predictor_uses_unified_prompt_for_both_profiles(self):
        for profile in ('ovobench', 'streamingbench'):
            engine = predict.Predictor.__new__(predict.Predictor)
            engine.args = self.args('--profile', profile)
            engine.model = engine.processor = None
            engine.torch = SimpleNamespace(no_grad=nullcontext, cuda=SimpleNamespace(empty_cache=lambda: None))
            engine.original_prompt = lambda *a, **kw: 'OLD_SINGLE_CHOICE_PROMPT'
            engine.ns = {'extract_frame': lambda *a: None}
            def infer(*a, **kw):
                builder = engine.ns['build_prompt']
                prompt = (builder(context_Scene_Graphs=None) if profile == 'ovobench'
                          else builder(kw['question'], kw['options'], None))
                return {'prompt': prompt, 'answer': 'A,B'}
            engine.ns['infer'] = infer
            for row in (self.contextual_row(), dict(video_path='missing', query_time=2,
                                                   question='Which?', options=['a', 'b', 'c', 'd'])):
                plan = predict.prepare(row, engine.args)
                result = engine.predict(plan, 1)
                self.assertEqual(result['prompt'], plan['prompt'])
                self.assertIn('Do not select additional options merely because you are uncertain.', result['prompt'])

    def test_qwen25_features(self):
        import torch
        image=torch.ones(4,8)
        model=SimpleNamespace(get_image_features=lambda *a:(image,))
        result=core.encode_qwen25(model,None,None)
        self.assertIs(result[0][0],image)
        self.assertIsNone(result[1]); self.assertEqual(result[2],4)

    def test_core_real_tensor_flow(self):
        import torch
        # Exercise the actual upstream statements and graph functions on CPU with toy models.
        utils=ModuleType('qwen_vl_utils'); utils.process_vision_info=lambda *a:None
        cv2=ModuleType('cv2')
        with patch.dict(sys.modules,{'qwen_vl_utils':utils,'cv2':cv2}), patch.object(torch.cuda,'empty_cache'), patch.object(torch.cuda,'synchronize'):
            for profile in ('ovobench','streamingbench'):
                for family in ('qwen3','qwen25'):
                    ns=core.build_core(profile,family)
                    calls=[]; encodes=[]
                    class Model:
                        device='cpu'
                        def get_image_features(self,*a):
                            encodes.append(1)
                            count=len(a[1])
                            image=tuple(torch.ones(2,4) for _ in range(count))
                            return (image,[]) if family=='qwen3' else image
                        def get_input_embeddings(self):
                            return lambda ids:torch.ones((*ids.shape,4))
                        def generate(self,**kw):
                            calls.append(kw)
                            return torch.tensor([[1,2]])
                    class Tokenizer:
                        def get_vocab(self):
                            return {'<|vision_start|>': 10, '<|vision_end|>': 11}
                        def __call__(self,text,**kw):
                            return {'input_ids':[[1,2]] if isinstance(text,list) else [1,2]}
                    class Processor:
                        if family == 'qwen3':
                            vision_start_token='<vision>'; vision_end_token='</vision>'
                        tokenizer=Tokenizer()
                        def image_processor(self,frames):
                            count=len(frames) if isinstance(frames,list) else 1
                            return dict(pixel_values=torch.ones(count*2,4),
                                        image_grid_thw=torch.tensor([[1,2,2]]*count))
                        def batch_decode(self,*a,**kw):
                            return ['Scene Graph:\n1. [person, holds, cup]' if calls[-1]['max_new_tokens']==256 else 'A']
                    timestamps=[]
                    def extract(path,t): timestamps.append(t); return object()
                    ns['extract_frame']=extract
                    opts=['one','two'] if profile=='ovobench' else repr(['A. one','B. two','C. three','D. four'])
                    with torch.no_grad():
                        result=ns['infer'](SimpleNamespace(fps=1,frame_interval=1,vision_batch_size=2),Model(),Processor(),
                                           'video','Question',opts,2,'00:00:02','STU')
                    self.assertEqual(timestamps,[0,1,2]); self.assertEqual(len(encodes),2)
                    self.assertEqual(result['encoded_frames'],3); self.assertEqual(result['answer'],'A')
                    self.assertEqual(len(calls),2 if profile=='ovobench' else 3)
                    self.assertEqual(calls[-1]['max_new_tokens'],128)
                    self.assertFalse(calls[-1]['do_sample'])
                    self.assertNotIn('pixel_values',calls[-1])
                    self.assertEqual(calls[-1]['inputs_embeds'].shape,(1,22,4))
                    self.assertEqual(result['scene_graph']['scene_graph'],[['person','holds','cup']])

    def test_qwen25_missing_vision_aliases(self):
        import torch
        sys.path.insert(0, str(core.UPSTREAM))
        from qwen_online.utils import whole_current_frame_tokens
        texts = []
        class Tokenizer:
            def get_vocab(self):
                return {'<|vision_start|>': 10, '<|vision_end|>': 11}
            def __call__(self, text, **kw):
                texts.append(text)
                return {'input_ids': [1, 2]}
        processor = SimpleNamespace(tokenizer=Tokenizer())
        model = SimpleNamespace(device='cpu', get_input_embeddings=lambda: lambda ids: torch.ones(len(ids), 4))
        frames = (torch.zeros(3, 4),)
        actual = core.whole_qwen25_frame_tokens(processor, model, 1.5, frames)
        self.assertEqual(texts, ['<|im_start|>user\n<1.5 seconds><|vision_start|>',
                                 '<|vision_end|><|im_end|>\n'])
        expected = whole_current_frame_tokens(processor, model, 1.5, frames)
        self.assertTrue(torch.equal(actual, expected))
        broken = SimpleNamespace(tokenizer=SimpleNamespace(get_vocab=lambda: {}))
        with self.assertRaisesRegex(ValueError, 'missing required token'):
            core.whole_qwen25_frame_tokens(broken, model, 1.5, frames)
        self.assertFalse(hasattr(broken, 'vision_start_token'))

    def test_dry_run_resume_and_shards(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d); source=root/'in.json'; output=root/'out.jsonl'
            source.write_text(json.dumps([dict(source_id='duplicate',video_path='missing',question='Q',query_time=q,options=['a','b']) for q in (700,2,350)]))
            common=['--input',str(source),'--output',str(output),'--model-path','/weights','--dry-run','--profile','streamingbench']
            self.assertEqual(predict.run(predict.parse_args(common)),0)
            rows=predict.read_rows(output)
            self.assertEqual([r['response_g1_prediction']['frame_interval'] for r in rows],[5,5,2])
            self.assertEqual(predict.run(predict.parse_args(common+['--resume'])),0)
            self.assertEqual(len(predict.read_rows(output)),3)
            with self.assertRaises(ValueError): predict.run(predict.parse_args(common+['--resume','--fps','2']))
            merged=root/'merged.jsonl'
            proc=subprocess.run([sys.executable,str(core.HERE/'run_multigpu.py'),'--gpus','0,1','--input',str(source),
                                 '--output',str(merged),'--model-path','/weights','--dry-run'],capture_output=True,text=True)
            self.assertEqual(proc.returncode,0,proc.stderr)
            self.assertEqual([r['response_g1_prediction']['input_index'] for r in predict.read_rows(merged)],[0,1,2])

    def test_scoring_uses_new_field(self):
        row=dict(question='Q',options=['a','b'],answer='B',prediction={'status':'completed','answer':'B'},
                 response_g1_prediction={'input_index':0,'status':'completed','answer':'A'})
        report,_=evaluate.evaluate([row]); self.assertEqual(report['overall']['correct'],0)
        row['response_g1_prediction']['status']='dry_run'
        with self.assertRaises(ValueError): evaluate.evaluate([row])


if __name__=='__main__': unittest.main()
