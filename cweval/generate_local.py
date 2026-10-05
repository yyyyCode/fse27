import datetime
import os
import re
from typing import Dict, List
import fire
import torch
from natsort import natsorted
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from cweval.commons import BENCHMARK_DIR, LANGS
from cweval.ppt import make_prompt

class Gener:
    begin_prompt_anchor = 'BEGIN PROMPT'
    begin_solution_anchor = 'BEGIN SOLUTION'
    def __init__(
        self,
        eval_path: str = '',
        model: str = '',
        ppt: str = 'direct',
        langs: List[str] = LANGS,
        exclude_path: List[str] = [],
        include_path: List[str] = [],
        n: int = 1,
        max_completion_tokens: int = 2048,
        temperature: float = 0.0,
        gpu_id: int = 2,
        **kwargs,
    ):
        self.model_path = model
        self.ppt = ppt
        self.langs = langs
        self.exclude_path = exclude_path
        self.include_path = include_path
        self.max_completion_tokens = max_completion_tokens
        self.gpu_id = gpu_id
        if n != 1:
            raise ValueError(
                'Greedy decoding requires n=1.'
            )
        if temperature != 0:
            raise ValueError(
                'Greedy decoding requires temperature=0.'
            )
        print(f'Using langs: {self.langs}')
        print(f'Using model: {self.model_path}')
        print('Decoding: greedy (do_sample=False)')
        if not torch.cuda.is_available():
            raise RuntimeError(
                'CUDA is not available. '
                'Please check PyTorch/CUDA/driver compatibility.'
            )
        gpu_count = torch.cuda.device_count()
        if gpu_id < 0 or gpu_id >= gpu_count:
            raise ValueError(
                f'Invalid gpu_id={gpu_id}. '
                f'Available GPU IDs: 0-{gpu_count - 1}'
            )
        self.device = torch.device(
            f'cuda:{gpu_id}'
        )
        print(f'Using GPU ID: {gpu_id}')
        print(f'Using device: {self.device}')
        print(
            f'GPU name: '
            f'{torch.cuda.get_device_name(gpu_id)}'
        )
        print('Loading tokenizer...')
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path,
            trust_remote_code=True,
        )
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = (
                self.tokenizer.eos_token_id
            )
        print('Loading model...')
        self.model = (
            AutoModelForCausalLM.from_pretrained(
                self.model_path,
                torch_dtype='auto',
                trust_remote_code=True,
            )
        )
        self.model.to(self.device)
        self.model.eval()
        print('Model loaded successfully.')
        if not eval_path:
            self.eval_path = os.path.join(
                'evals',
                'eval_'
                + datetime.datetime.now().strftime(
                    '%y%m%d_%H%M%S'
                ),
            )
        else:
            if os.path.exists(eval_path):
                flag = (
                    input(
                        f'{eval_path} already exists, '
                        f'continue? (y/n): '
                    )
                    .strip()
                    .lower()
                )
                if flag != 'y':
                    print('Exiting...')
                    exit(0)
            self.eval_path = eval_path
        self.cases = self._get_cases()
    def _get_cases(
        self,
    ) -> Dict[str, Dict[str, str]]:
        cases = {}
        for root, _, files in os.walk(
            BENCHMARK_DIR
        ):
            if '__pycache__' in root:
                continue
            for file in natsorted(files):

                file_wo_ext, ext = (
                    os.path.splitext(file)
                )
                task_file_path = os.path.join(
                    root,
                    file,
                )
                lang = ext[1:]
                if not (
                    ext
                    and file_wo_ext.endswith(
                        '_task'
                    )
                ):
                    continue
                if lang not in self.langs:
                    continue
                if any(
                    exclude in task_file_path
                    for exclude
                    in self.exclude_path
                ):
                    continue
                if (
                    self.include_path
                    and not any(
                        include in task_file_path
                        for include
                        in self.include_path
                    )
                ):
                    continue
                with open(
                    task_file_path,
                    'r',
                ) as f:
                    task_code = f.read()
                begin_solution_line_src = ''
                for line in task_code.splitlines():
                    if (
                        self.begin_solution_anchor
                        in line
                    ):
                        begin_solution_line_src = (
                            line
                        )
                        break
                if not begin_solution_line_src:
                    raise ValueError(
                        'No solution found in '
                        f'{task_file_path}'
                    )
                code_prompt = (
                    task_code
                    .split(
                        self.begin_prompt_anchor
                    )[-1]
                    .split(
                        begin_solution_line_src
                    )[0]
                    .strip()
                )
                rel_task_file_path = (
                    os.path.relpath(
                        task_file_path,
                        BENCHMARK_DIR,
                    )
                )
                gen_file_path_template = (
                    os.path.join(
                        self.eval_path,
                        'generated_{index}',
                        rel_task_file_path.replace(
                            '_task',
                            '_raw',
                        ),
                    )
                )
                cases[task_file_path] = {
                    'task_file_path':
                        task_file_path,
                    'code_prompt':
                        code_prompt,
                    'lang':
                        lang,
                    'out_path_template':
                        gen_file_path_template,
                }

        return cases
    @staticmethod
    def _extract_code(
        response: str,
    ) -> str:
        response = response.strip()
        match = re.search(
            r'```[^\n]*\n(.*?)```',
            response,
            flags=re.DOTALL,
        )
        if match:
            return match.group(1).strip()
        if response.startswith('```'):
            lines = response.splitlines()
            lines = lines[1:]

            if (
                lines
                and lines[-1].strip() == '```'
            ):
                lines = lines[:-1]

            return '\n'.join(lines).strip()
        return response
    def _gen_case(
        self,
        case: Dict[str, str],
    ) -> None:
        out_path = (
            case['out_path_template']
            .format(index=0)
        )
        if os.path.exists(out_path):

            print(
                f'{out_path} already completed, '
                f'skipping'
            )
            return
        prompt = make_prompt(
            self.ppt
        )
        prompt_text = prompt.PPT.format(
            lang=case['lang'],
            lang_instr=(
                prompt.LANG_INSTR[
                    case['lang']
                ]
            ),
            code_prompt=case[
                'code_prompt'
            ],
        )
        messages = [
            {
                'role': 'user',
                'content': prompt_text,
            }
        ]
        if (
            self.tokenizer.chat_template
            is not None
        ):
            try:
                text = (
                    self.tokenizer
                    .apply_chat_template(
                        messages,
                        tokenize=False,
                        add_generation_prompt=True,
                        enable_thinking=False,
                    )
                )
            except TypeError:

                text = (
                    self.tokenizer
                    .apply_chat_template(
                        messages,
                        tokenize=False,
                        add_generation_prompt=True,
                    )
                )
        else:
            text = prompt_text
        inputs = self.tokenizer(
            text,
            return_tensors='pt',
        ).to(self.device)

        input_length = (
            inputs['input_ids'].shape[1]
        )
        with torch.inference_mode():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=(
                    self.max_completion_tokens
                ),
                do_sample=False,
                num_beams=1,
                pad_token_id=(
                    self.tokenizer.pad_token_id
                ),
                eos_token_id=(
                    self.tokenizer.eos_token_id
                ),
                use_cache=True,
            )
        generated_ids = outputs[
            0
        ][input_length:]
        response = (
            self.tokenizer.decode(
                generated_ids,
                skip_special_tokens=True,
            )
            .strip()
        )
        response = self._extract_code(
            response
        )
        os.makedirs(
            os.path.dirname(out_path),
            exist_ok=True,
        )
        with open(
            out_path,
            'w',
        ) as f:
            f.write(response)
    def gen(self) -> None:
        print(
            f'Number of cases: '
            f'{len(self.cases)}'
        )
        for case in tqdm(
            self.cases.values(),
            total=len(self.cases),
        ):
            try:
                self._gen_case(
                    case
                )
            except Exception as e:
                print(
                    'Error in '
                    f'{case["task_file_path"]}: '
                    f'{e}',
                    flush=True,
                )
if __name__ == '__main__':
    fire.Fire(Gener)