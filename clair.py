import os
import json
from tqdm import tqdm
import argparse

from openai import AuthenticationError, OpenAI
import json
import logging
import re
import sys
from typing import List, Optional, Tuple, Any
import random

_engine_cache = {}


class LLMEngine:
    @staticmethod
    def from_string(model_name):
        if model_name == "gpt-4o":
            return OpenAIEngine(model_name="gpt-4o")
        elif model_name == "chat-gpt":
            return OpenAIEngine(model_name="gpt-3.5-turbo")
        else:
            raise ValueError(f"Unknown model: {model_name}")

    def __call__(self, prompt: str, n_completions: int = 1, **kwargs: Any) -> List[str]:
        return self.call(prompt, n_completions, **kwargs)


class OpenAIEngine(LLMEngine):
    def __init__(self, model_name):
        self.model_name = model_name
        self.api = OpenAI()

    def call(self, prompt, n_completions, temperature=0.0, max_tokens=1024):
        messages = [{"role": "user", "content": prompt}]
        response = self.api.chat.completions.create(
            model=self.model_name,
            messages=messages,
            n=n_completions,
            temperature=temperature,
            max_tokens=max_tokens,
        )

        return [r.message.content for r in response.choices]


_CLAIR_PROMPT = """\
You are trying to tell if a candidate set of captions is describing the same image as a reference set of captions.
Candidate set:
{candidate_statements}
Reference set:
{target_statements}
On a precise scale from 0 to 100, how likely is it that the candidate set is \
describing the same image as the reference set? (JSON format, with a key "score", \
value between 0 and 100, and a key "reason" with a string value.)
"""


def clair(
    candidates: List[str],
    targets: List[str],
    model: str = "chat-gpt",
    max_tokens: int = 1024,
) -> Tuple[float, Optional[str]]:
    # Compute the CLAIR score for a list of candidates and targets.

    if model not in _engine_cache:
        _engine_cache[model] = LLMEngine.from_string(model)

    # Format the canndidates and targets
    candidate_statements = [f"- {c}\n" for c in candidates]
    target_statements = [f"- {t}\n" for t in targets]
    formatted_prompt = _CLAIR_PROMPT.format(
        candidate_statements="".join(candidate_statements),
        target_statements="".join(target_statements),
    )

    temperature, score, reason = 0.0, None, None
    for _ in range(3):
        # Run the model
        logging.debug(f'CLAIR prompt: "{formatted_prompt}"')
        response = _engine_cache[model](
            formatted_prompt, temperature=temperature, max_tokens=max_tokens
        )[0]
        logging.debug(f'CLAIR response: "{response.strip()}"')

        # Parse the first JSON object in the response
        try:
            parsed = response.split("{")[1]
            parsed = "{" + parsed.split("}")[0] + "}"
            data = json.loads(parsed)
            score = float(data["score"])
            reason = data.get("reason", "Unknown")
            break
        except (json.JSONDecodeError, KeyError, IndexError):
            # Try to extract the first number in the response using regex
            parsed = re.findall(r"\d*\.?\d+", response)
            if len(parsed) > 0:
                score = float(parsed[0])
                if score < 1:
                    score *= 100  # This is a weird situation where some models auto-normalize the score for us.

                # Look for the word "reason" in the response, and extract anything after it (ignoring case)
                reason = re.findall(r"(?i)reason.*", response)
                if len(reason) > 0:
                    # Clean up the reason a bit.
                    reason = reason[0].strip()[len("reason") :].replace(":", "").strip()
                else:
                    reason = "Unknown"
                break
            else:
                logging.warn(
                    f"Could not parse response from CLAIR: {response}. Retrying"
                )
                continue
    else:
        logging.error(
            "Could not parse response from CLAIR after 3 tries. Setting score to 0."
        )
        score = 0.0
        reason = None

    return score / 100, reason


def eval_model(args):
    assert args.data_type in ["iiw", "docci"]
    exp_name = args.experiment_name
    answer_file = os.path.join(args.answer_folder, exp_name + ".jsonl")
    save_path = os.path.join(args.save_folder, exp_name + "_clair.jsonl")
    annotation_file = args.annotation_file

    os.environ["OPENAI_API_KEY"] = args.openai_api_key
    answers = [json.loads(q) for q in open(os.path.expanduser(answer_file), "r")]

    annotations = [
        json.loads(q) for q in open(os.path.expanduser(annotation_file), "r")
    ]

    if args.data_type == "docci":
        rng = random.Random(args.seed)
        annotations = rng.sample(annotations, 500)

    directory = os.path.dirname(save_path)

    if not os.path.exists(directory):
        os.makedirs(directory)

    save_file = open(save_path, "w")

    clairs = []
    for answer, annotation in tqdm(zip(answers, annotations), total=len(answers)):
        save_output = {}
        if args.data_type == "iiw":
            references = [annotation["IIW"]]
        elif args.data_type == "docci":
            references = [annotation["description"]]
        candidates = [answer["caption"]]

        clair_score = clair(candidates, references, model="gpt-4o")

        # print(f"ref: {references[0]}")
        # print(f"cand: {candidates[0]}")
        # print(clair_score)

        clairs.append(clair_score)
        save_output["score"] = clair_score[0]
        save_output["reason"] = clair_score[1]

        save_file.write(json.dumps(save_output) + "\n")
    save_file.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--annotation_file",
        type=str,
        default="/home/mingi/experiments/LLaVA/data/eval/imageinwords/IIW-400/data.jsonl",
    )
    parser.add_argument(
        "--answer_folder",
        type=str,
        default="/home/mingi/experiments/LLaVA/data/experiments/iiw/",
    )
    parser.add_argument(
        "--save_folder",
        type=str,
        default="/home/mingi/experiments/SPARC/results",
    )
    parser.add_argument("--experiment_name", type=str, default=None)
    parser.add_argument("--openai_api_key", type=str, default=None)
    parser.add_argument("--data_type", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    eval_model(args)
