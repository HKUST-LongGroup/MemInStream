import contextlib
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import predict as p
from prompt_templates import make_prompt


def args(*extra):
    return p.parse_args(['--input','input.json','--output','out.json','--model-path','missing',*extra])


def row(**extra):
    return dict(video_path='missing.mp4', query_time=100, question='Which objects?',
                options=['A. Cup','B. Plate'], correct_option=['A','B'], answer='SECRET',
                source_dataset='test', source_id='1', **extra)


class AdapterTests(unittest.TestCase):
    def test_source_and_reference_checksums(self):
        self.assertEqual(len(p.verify_upstream()),40)
        for name, ref in json.loads((p.HERE/'reference_manifest.json').read_text()).items():
            self.assertEqual(hashlib.sha256((p.HERE/name).read_bytes()).hexdigest(),ref['sha256'])

    def test_official_defaults_and_fps_boundaries(self):
        a=args()
        self.assertEqual((a.history_seconds,a.min_pixels,a.max_pixels,a.min_frames,a.max_frames,
                          a.max_new_tokens,a.drop_method,a.drop_threshold,a.drop_relative),
                         (0,448*448,448*448,4,1016,128,'feature',0.5,False))
        self.assertEqual([p.sampling_fps(a,t) for t in [300,300.001,600,600.001]], [1,.5,.5,.2])
        self.assertEqual(p.sampling_fps(args('--fps','4'),1000),4)
        self.assertEqual(p.window(row(),30),(70,100))
        self.assertEqual(p.window(row(),30,90),(70,90))
        with self.assertRaises(ValueError): p.window(row(),30,60)

    def test_prompts_history_and_gold_leakage(self):
        r=row(dialogue_events=[{'time':90,'role':'assistant','content':'OLD'},
                               {'time':100,'role':'assistant','content':'FUTURE'}])
        a=args('--dry-run','--history-seconds','30')
        result=p.predict_row(0,r,a,None)
        self.assertEqual(result['prediction']['prompt'],make_prompt(r,70,100))
        self.assertIn('OLD',result['prediction']['prompt'])
        self.assertNotIn('SECRET',result['prediction']['prompt'])
        self.assertNotIn('FUTURE',result['prediction']['prompt'])
        self.assertEqual(result['answer'],'SECRET')
        self.assertEqual(len(p.build_messages('EXACT','clip',args(),1)),1)
        messages=p.build_messages('EXACT','clip',args('--system-prompt','qwen'),1)
        self.assertEqual(messages[0]['role'],'system')
        self.assertEqual(messages[1]['content'][1]['text'],'EXACT')

    def test_failure_rows_are_retained(self):
        r=row();r['query_time']=0
        out=p.predict_row(7,r,args('--dry-run'),None)
        self.assertEqual(out['prediction']['status'],'error')
        self.assertEqual(out['prediction']['row_index'],7)
        self.assertEqual(out['source_id'],'1')

    def test_cli_parallel_jsonl_and_no_overwrite(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); src=root/'input.json';out=root/'output.jsonl'
            src.write_text(json.dumps([row(),dict(row(),source_id='2')]))
            cmd=[sys.executable,str(p.HERE/'predict.py'),'--input',str(src),'--output',str(out),
                 '--model-path','missing','--dry-run','--num-gpus','2']
            run=subprocess.run(cmd,capture_output=True,text=True,timeout=60)
            self.assertEqual(run.returncode,0,run.stderr)
            records=[json.loads(l) for l in out.read_text().splitlines()]
            self.assertEqual([x['source_id'] for x in records],['1','2'])
            self.assertTrue(all(x['prediction']['status']=='dry_run' for x in records))
            self.assertNotEqual(subprocess.run(cmd,capture_output=True,timeout=60).returncode,0)

    def test_generate_forwards_official_parameters_and_trims_prompt(self):
        class Tensor(list):
            @property
            def shape(self): return (len(self),len(self[0]))
            def tolist(self): return list(self)
        class Inputs(dict):
            def __getattr__(self,name): return self[name]
            def to(self,device): return self
        class Processor:
            def apply_chat_template(self,messages,**kw):
                self.messages=messages;return 'CHAT'
            def __call__(self,**kw):
                self.kw=kw
                return Inputs(input_ids=Tensor([[1,2,3]]),video_grid_thw=Tensor([[2,32,32]]))
            def batch_decode(self,tokens,**kw):
                assert tokens==[[8,9]],tokens
                return ['A,B']
        class Model:
            device='cuda'
            def generate(self,**kw): self.kw=kw;return [[1,2,3,8,9]]
        e=p.Predictor.__new__(p.Predictor);e.args=args();e.processor=Processor();e.model=Model()
        e.torch=SimpleNamespace(inference_mode=contextlib.nullcontext)
        e.commit=p.verify_upstream();e.generation_config={'do_sample':True};e.model_provenance={}
        e.process_vision_info=lambda messages:(None,[SimpleNamespace(shape=(4,3,448,448))])
        with patch.object(p,'clip_video'):
            answer,stats=e.generate('EXACT',dict(path='v',start=0,end=10,fps=1,drop_path='drop'))
        self.assertEqual(answer,'A,B')
        self.assertEqual(stats['input_tokens'],3)
        for k,v in dict(max_new_tokens=128,drop_method='feature',drop_threshold=.5,drop_absolute=True,
                        dr_save_path='drop').items(): self.assertEqual(e.model.kw[k],v)
        self.assertNotIn('do_sample',e.model.kw)
        self.assertNotIn('fps',e.processor.kw)
        self.assertNotIn('video_kwargs',e.processor.kw)

    @unittest.skipUnless(shutil.which('ffmpeg') and shutil.which('ffprobe'),'ffmpeg unavailable')
    def test_fractional_clip_contains_no_future_frames(self):
        for binary in ['ffmpeg','ffprobe']:
            check=subprocess.run([binary,'-version'],capture_output=True)
            if check.returncode:
                self.skipTest(f'{binary} executable is broken: {check.stderr.decode(errors="replace")[:180]}')
        # Source: red until t=1.0, blue thereafter. Query at 0.9 must see only red.
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);video=root/'source.mp4';clip=root/'clip.mp4'
            subprocess.run(['ffmpeg','-v','error','-f','lavfi','-i','color=red:s=64x64:r=10:d=1',
                '-f','lavfi','-i','color=blue:s=64x64:r=10:d=1','-filter_complex',
                '[0:v][1:v]concat=n=2:v=1:a=0','-c:v','libx264','-y',str(video)],check=True)
            p.clip_video(video,.2,.9,clip,args())
            raw=subprocess.run(['ffmpeg','-v','error','-i',str(clip),'-pix_fmt','rgb24','-f','rawvideo','pipe:1'],
                               capture_output=True,check=True).stdout
            self.assertTrue(raw)
            self.assertGreater(sum(raw[0::3])/len(raw[0::3]),200)
            self.assertLess(sum(raw[2::3])/len(raw[2::3]),20)
            self.assertAlmostEqual(p.probe_video(str(clip),30)[2],.7,places=2)


if __name__=='__main__': unittest.main()
