#!/usr/bin/python3

from tqdm import tqdm
from pathlib import Path
from typing import List, Tuple, Dict, Any
from .question_model import QAgent
import random
import json


class QuestioningAgent(object):
    """Agent for generating logical reasoning questions"""

    def __init__(self, **kwargs):
        self.agent = QAgent(**kwargs)
        random.seed(42)

    def build_prompt(self, topic: str) -> Tuple[str, str]:
        """Generate prompt for question generation"""

        sys_prompt = """You are Q-Agent, an expert question generator for logical reasoning.

ALLOWED TOPICS:
1. blood_relations - Family relationship puzzles
2. seating_arrangement - Linear row or circular table seating

REQUIREMENTS:
- 3-6 named entities per question
- At least 3 unique constraints/relations
- Self-contained (all info in question text)
- No coded relations (use plain English)
- Exactly 4 choices: ["A) ...", "B) ...", "C) ...", "D) ..."]
- Single letter answer: "A", "B", "C", or "D"
- Brief explanation (under 100 words)
- Step-by-step reasoning (5 steps as single string)

OUTPUT FORMAT (NO difficulty field):
{
  "topic": "blood_relations" or "seating_arrangement",
  "question": "Complete self-contained problem with all information",
  "choices": ["A) option1", "B) option2", "C) option3", "D) option4"],
  "answer": "A" or "B" or "C" or "D",
  "explanation": "Brief justification under 100 words",
  "reasoning": "Step 1: ... Step 2: ... Step 3: ... Step 4: ... Step 5: ..."
}

Return ONLY valid JSON, no markdown blocks."""

        topic_readable = topic.split("/")[-1].replace("_", " ")
        simple_prompt = f"Generate a {topic_readable} question with 3-6 entities, multiple choice options, and step-by-step reasoning."

        return simple_prompt, sys_prompt

    def generate_question(
        self,
        topic: Tuple[str, str] | List[Tuple[str, str]],
        **gen_kwargs,
    ) -> Tuple[List[str], int | None, float | None]:
        """Generate question(s) from topic(s)"""
        if isinstance(topic, list):
            prompts = []
            sp = None
            for t in topic:
                p, sp_temp = self.build_prompt(f"{t[0]}/{t[1]}")
                prompts.append(p)
                if sp is None:
                    sp = sp_temp
            prompt = prompts
        else:
            prompt, sp = self.build_prompt(f"{topic[0]}/{topic[1]}")

        resp, tl, gt = self.agent.generate_response(prompt, sp, **gen_kwargs)

        if isinstance(resp, str):
            return [resp], tl, gt
        elif isinstance(resp, list):
            return resp, tl, gt
        else:
            return [""] * (len(topic) if isinstance(topic, list) else 1), tl, gt

    def generate_batches(
        self,
        num_questions: int,
        topics: Dict[str, List[str]],
        batch_size: int = 5,
        **kwargs,
    ) -> Tuple[List[str], List[int | None], List[float | None]]:
        """Generate questions in batches"""
        extended_topics = self.populate_topics(topics, num_questions)
        questions = []
        tls, gts = [], []
        total_batches = (len(extended_topics) + batch_size - 1) // batch_size
        pbar = tqdm(total=total_batches, desc="Generating")

        for i in range(0, len(extended_topics), batch_size):
            batch_topics = extended_topics[i : i + batch_size]
            batch_questions, tl, gt = self.generate_question(batch_topics, **kwargs)
            questions.extend(batch_questions)
            tls.append(tl)
            gts.append(gt)
            pbar.update(1)

        pbar.close()
        return questions, tls, gts

    def populate_topics(
        self, topics: Dict[str, List[str]], num_questions: int
    ) -> List[Tuple[str, str]]:
        """Randomly select topics for generation"""
        all_subtopics = [(t, st) for t, sublist in topics.items() for st in sublist]
        if not all_subtopics:
            raise ValueError("No subtopics found")
        return random.choices(all_subtopics, k=num_questions)

    def filter_questions(
        self, questions: List[str | Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Filter valid questions"""
        def is_valid(q: Dict[str, str]) -> bool:
            required = ["topic", "question", "choices", "answer"]
            if not all(key in q for key in required):
                return False

            choices = q["choices"]
            if not isinstance(choices, list) or len(choices) != 4:
                return False

            if not all(isinstance(c, str) and len(c) > 2 and c[0] in "ABCD" for c in choices):
                return False

            if not isinstance(q["answer"], str):
                return False

            return True

        valid = []
        for q in questions:
            if isinstance(q, dict):
                if is_valid(q):
                    valid.append(q)
            elif isinstance(q, str):
                try:
                    q_dict = json.loads(q)
                    if is_valid(q_dict):
                        valid.append(q_dict)
                except json.JSONDecodeError:
                    continue

        return valid if len(valid) >= 0.5 * len(questions) else []

    def save_questions(self, questions: Any, file_path: str | Path) -> None:
        """Save questions to JSON file"""
        file_path = Path(file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        with open(file_path, "w") as f:
            json.dump(questions, f, indent=4)

    @staticmethod
    def load_icl_samples(file_path: str | Path) -> Dict[str, List[Dict[str, str]]]:
        """Load in-context learning samples"""
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"File {file_path} does not exist")
        with open(file_path, "r") as f:
            return json.load(f)


if __name__ == "__main__":
    import argparse
    import yaml
    import os

    argparser = argparse.ArgumentParser(description="Generate logical reasoning questions")
    argparser.add_argument("--num_questions", type=int, default=10, help="Number of questions")
    argparser.add_argument("--output_file", type=str, default="outputs/questions.json", help="Output file")
    argparser.add_argument("--batch_size", type=int, default=5, help="Batch size")
    argparser.add_argument("--verbose", action="store_true", help="Verbose output")
    args = argparser.parse_args()

    assets_dir = "assets" if os.path.exists("assets/topics.json") else "assets_v1"

    with open(f"{assets_dir}/topics.json") as f:
        topics = json.load(f)

    agent = QuestioningAgent()

    gen_kwargs = {"tgps_show": True}
    with open("qgen.yaml", "r") as f:
        gen_kwargs.update(yaml.safe_load(f))

    questions, tls, gts = agent.generate_batches(
        num_questions=args.num_questions,
        topics=topics,
        batch_size=args.batch_size,
        **gen_kwargs,
    )

    if args.verbose:
        print(f"\nGenerated {len(questions)} questions")
        if gen_kwargs.get("tgps_show"):
            print(f"Time: {sum(gts):.3f}s | Tokens: {sum(tls)} | TGPS: {sum(tls)/sum(gts):.3f}")

    agent.save_questions(questions, args.output_file)
    filtered = agent.filter_questions(questions)
    filtered_file = args.output_file.replace("questions.json", "filtered_questions.json")
    agent.save_questions(filtered, filtered_file)

    if args.verbose:
        print(f"Saved to {args.output_file}")
