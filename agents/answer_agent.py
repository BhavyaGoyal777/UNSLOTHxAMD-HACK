#!/usr/bin/python3

import re
import json
from pathlib import Path
from tqdm import tqdm
from typing import List, Tuple, Dict, Any
from .answer_model import AAgent


class AnsweringAgent(object):
    """Agent for solving logical reasoning questions"""

    def __init__(self, **kwargs):
        self.agent = AAgent(**kwargs)

    def build_prompt(self, question_data: Dict[str, Any]) -> Tuple[str, str]:
        """Generate prompt for answering question"""

        sys_prompt = """You are A-Agent, an expert logical reasoning solver.

SOLVING STRATEGY:

Blood Relations:
1. Identify speaker perspective
2. Resolve each relation step-by-step
3. Maintain generational levels
4. Track gender (assume male if not stated)
5. Finalize relationship

Seating Arrangement:
1. Identify type (linear/circular)
2. Determine facing direction (if circular)
3. List all constraints
4. Apply constraints step-by-step
5. Verify final arrangement

OUTPUT FORMAT:

answer: "A" or "B" or "C" or "D"
reasoning: "Step 1: [description]. Step 2: [description]. Step 3: [description]. Step 4: [description]. Step 5: [description]."

REQUIREMENTS:
- answer must be single capital letter A, B, C, or D
- reasoning must be exactly 5 steps in single string
- Each step starts with "Step N:"
- All steps in ONE continuous string, separated by periods"""

        formatted = f"{question_data['question']}\n\nChoices:\n"
        for choice in question_data['choices']:
            formatted += f"{choice}\n"
        prompt = formatted.strip()

        return prompt, sys_prompt

    def answer_question(
        self, question_data: Dict | List[Dict], **kwargs
    ) -> Tuple[List[str], int | None, float | None]:
        """Generate answer(s) for question(s)"""
        if isinstance(question_data, list):
            prompts = []
            sp = None
            for qd in question_data:
                p, sp_temp = self.build_prompt(qd)
                prompts.append(p)
                if sp is None:
                    sp = sp_temp
            prompt = prompts
        else:
            prompt, sp = self.build_prompt(question_data)

        resp, tl, gt = self.agent.generate_response(prompt, sp, **kwargs)

        if isinstance(resp, str):
            return [resp], tl, gt
        elif isinstance(resp, list):
            return resp, tl, gt
        else:
            return [""] * (len(question_data) if isinstance(question_data, list) else 1), tl, gt

    def answer_batches(
        self, questions: List[Dict], batch_size: int = 5, **kwargs
    ) -> Tuple[List[str], List[int | None], List[float | None]]:
        """Answer questions in batches"""
        answers = []
        tls, gts = [], []
        total_batches = (len(questions) + batch_size - 1) // batch_size
        pbar = tqdm(total=total_batches, desc="Answering")

        for i in range(0, len(questions), batch_size):
            batch_questions = questions[i : i + batch_size]
            batch_answers, tl, gt = self.answer_question(batch_questions, **kwargs)
            answers.extend(batch_answers)
            tls.append(tl)
            gts.append(gt)
            pbar.update(1)

        pbar.close()
        return answers, tls, gts

    def parse_answer(self, text: str) -> Dict[str, str]:
        """Parse answer from model output"""
        answer_match = re.search(r'answer:\s*"([A-D])"', text)
        reasoning_match = re.search(r'reasoning:\s*"([^"]+)"', text)

        if answer_match and reasoning_match:
            return {
                "answer": answer_match.group(1),
                "reasoning": reasoning_match.group(1)
            }
        return {"answer": "", "reasoning": ""}

    def save_answers(self, answers: List[Dict], file_path: str | Path) -> None:
        """Save answers to JSON file"""
        file_path = Path(file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        with open(file_path, "w") as f:
            json.dump(answers, f, indent=4)


if __name__ == "__main__":
    import yaml
    import argparse

    argparser = argparse.ArgumentParser(description="Answer logical reasoning questions")
    argparser.add_argument("--input_file", type=str, default="outputs/filtered_questions.json", help="Input questions file")
    argparser.add_argument("--output_file", type=str, default="outputs/answers.json", help="Output answers file")
    argparser.add_argument("--batch_size", type=int, default=5, help="Batch size")
    argparser.add_argument("--verbose", action="store_true", help="Verbose output")
    args = argparser.parse_args()

    with open(args.input_file, "r") as f:
        questions = json.load(f)

    agent = AnsweringAgent()

    gen_kwargs = {"tgps_show": True}
    with open("agen.yaml", "r") as f:
        gen_kwargs.update(yaml.safe_load(f))

    raw_answers, tls, gts = agent.answer_batches(
        questions=questions,
        batch_size=args.batch_size,
        **gen_kwargs
    )

    parsed_answers = [agent.parse_answer(a) for a in raw_answers]

    if args.verbose:
        print(f"\nAnswered {len(parsed_answers)} questions")
        if gen_kwargs.get("tgps_show"):
            print(f"Time: {sum(gts):.3f}s | Tokens: {sum(tls)} | TGPS: {sum(tls)/sum(gts):.3f}")

    agent.save_answers(parsed_answers, args.output_file)

    if args.verbose:
        print(f"Saved to {args.output_file}")
