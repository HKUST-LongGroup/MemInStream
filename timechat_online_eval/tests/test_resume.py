import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import predict as p

class ResumeTests(unittest.TestCase):
    def test_resume_out_of_order_legacy_progress_and_partial_line(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);src=root/'input.json';out=root/'output.jsonl'
            rows=[dict(source_id=str(i),video_path='missing',query_time=100,question='Which?',options=['A','B']) for i in range(5)]
            src.write_text(json.dumps(rows))
            cli=['--input',str(src),'--output',str(out),'--model-path','missing','--dry-run','--num-gpus','2']
            args=p.parse_args(cli)
            saved=[p.predict_row(i,rows[i],args,None) for i in [2,0]]
            journal=out.with_suffix('.progress.jsonl')
            journal.write_text(''.join(json.dumps(r)+'\n' for r in saved)+'{"prediction":')
            command=[sys.executable,str(p.HERE/'predict.py'),*cli,'--resume']
            run=subprocess.run(command,capture_output=True,text=True,timeout=60)
            self.assertEqual(run.returncode,0,run.stderr)
            results=[json.loads(x) for x in out.read_text().splitlines()]
            self.assertEqual([r['prediction']['row_index'] for r in results],list(range(5)))
            self.assertEqual(results[2],saved[0])
            self.assertEqual(len(journal.read_text().splitlines()),5)
            # A second resume is a no-op with no model load or duplicate records.
            self.assertEqual(subprocess.run(command,capture_output=True,timeout=60).returncode,0)
            self.assertEqual(len(journal.read_text().splitlines()),5)
            run=subprocess.run(command+['--fps','2'],capture_output=True,text=True,timeout=60)
            self.assertNotEqual(run.returncode,0)
            self.assertIn('configuration changed',run.stderr)
            rows[2]['question']='changed';src.write_text(json.dumps(rows))
            run=subprocess.run(command,capture_output=True,text=True,timeout=60)
            self.assertNotEqual(run.returncode,0)
            self.assertIn('Input changed',run.stderr)

    def test_error_is_preserved_and_interior_corruption_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);journal=root/'p.jsonl'
            args=p.parse_args(['--input','i','--output','o.jsonl','--model-path','m','--dry-run'])
            rows=[dict(video_path='v',query_time=0,question='q')]
            result=p.predict_row(0,rows[0],args,None)
            journal.write_text(json.dumps(result)+'\n')
            saved,_,_=p.read_progress(journal,rows,args)
            self.assertEqual(saved[0]['prediction']['status'],'error')
            journal.write_text('{bad}\n'+json.dumps(result)+'\n')
            with self.assertRaisesRegex(ValueError,'Corrupt'):
                p.read_progress(journal,rows,args)
