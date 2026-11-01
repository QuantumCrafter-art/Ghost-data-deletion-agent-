import os
import tempfile
import unittest
from pathlib import Path
import sys

CODE = Path(__file__).resolve().parents[1] / "code"
sys.path.insert(0, str(CODE))
from ghost_audit.envfile import load_project_env

class EnvFileTests(unittest.TestCase):
    def test_code_env_loads_without_printing_values(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1].parent) as temp:
            root=Path(temp); (root/"code").mkdir()
            (root/"code/.env").write_text("GEMINI_API_KEY=synthetic-test-key\nGEMINI_MODEL=gemini-test\n",encoding="utf-8")
            before={k:os.environ.get(k) for k in ("GEMINI_API_KEY","GEMINI_MODEL")}
            try:
                os.environ.pop("GEMINI_API_KEY",None); os.environ.pop("GEMINI_MODEL",None)
                loaded=load_project_env(root)
                self.assertEqual(set(loaded),{"GEMINI_API_KEY","GEMINI_MODEL"})
                self.assertEqual(os.environ["GEMINI_API_KEY"],"synthetic-test-key")
                self.assertEqual(os.environ["GEMINI_MODEL"],"gemini-test")
            finally:
                for key,value in before.items():
                    if value is None: os.environ.pop(key,None)
                    else: os.environ[key]=value

    def test_existing_process_variables_win_and_root_file_precedes_code_file(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1].parent) as temp:
            root=Path(temp); (root/"code").mkdir()
            (root/".env").write_text("KEEP=from-root\nROOT_ONLY=root\n",encoding="utf-8")
            (root/"code/.env").write_text("KEEP=from-code\nCODE_ONLY=code\n",encoding="utf-8")
            before={k:os.environ.get(k) for k in ("KEEP","ROOT_ONLY","CODE_ONLY")}
            try:
                os.environ["KEEP"]="from-process"
                os.environ.pop("ROOT_ONLY",None); os.environ.pop("CODE_ONLY",None)
                load_project_env(root)
                self.assertEqual(os.environ["KEEP"],"from-process")
                self.assertEqual(os.environ["ROOT_ONLY"],"root")
                self.assertEqual(os.environ["CODE_ONLY"],"code")
            finally:
                for key,value in before.items():
                    if value is None: os.environ.pop(key,None)
                    else: os.environ[key]=value

if __name__ == "__main__": unittest.main()
