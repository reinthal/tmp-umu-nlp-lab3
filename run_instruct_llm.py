from dataclasses import dataclass
from typing import List, Tuple

from openai import OpenAI
from tqdm import tqdm

prompt1_zero_shot = """
please classify this text as either neg or pos
reply with only those tokens.
"""

prompt2_one_shot = """
please classify this text as either neg or pos
reply with only those tokens.

input: oh man , this sucks really bad . good thing nu-metal is dead . thrash metal is real metal , this is for posers
output: neg
"""

prompt3_xml_one_shot = """
<SYSTEM>
please classify this text as either neg or pos
reply with only the tokens within <OUTPUT>YOURANSWER</OUTPUT> xml tags.
</SYSTEM>
<EXAMPLES>
<INPUT>
oh man , this sucks really bad . good thing nu-metal is dead . thrash metal is real metal , this is for posers
</INPUT>
<OUTPUT>
neg
</OUTPUT>
</EXAMPLES>
"""

prompt3_xml_few_shot = """
<SYSTEM>
please classify this text as either neg or pos
reply with only those tokens.

<EXAMPLES>
<INPUT>
oh man , this sucks really bad . good thing nu-metal is dead . thrash metal is real metal , this is for posers
</INPUT>
<OUTPUT>
neg
</OUTPUT>
<INPUT>
i can only say that this was my favorite record of last year . this record is hard to find in your local record store , so search it out.. . incredible - ca n't wait to hear more !
</INPUT>
<OUTPUT>
pos 
</OUTPUT>
<INPUT>
i was looking for a pair of roof-prism binoculars like this , for hawk and bird watching . i had looked at swarovskis , but they were way out of my budget , nearly a thousand dollars for this resolution . these binoculars are crystal-clear , with sharp focus . the 42mm lens lets in lots of light , lots more than the 35 's i 've been using . i wear glasses and the twist - up - and -down eyecups are a blessing . there are also attached lens protectors for the 42mm , nice because i always used to lose them . for the price , these ca n't be beat
</INPUT>
<OUTPUT>
neg 
</OUTPUT>
</EXAMPLES>
"""

prompt4_xml_few_shot = """
<SYSTEM>
please classify this text as either neg or pos
reply with only those tokens.

<EXAMPLES>
<INPUT>
oh man , this sucks really bad . good thing nu-metal is dead . thrash metal is real metal , this is for posers
</INPUT>
<OUTPUT>
neg
</OUTPUT>
<INPUT>
oh man , this sucks really bad . good thing nu-metal is dead . thrash metal is real metal , this is for posers
</INPUT>
<OUTPUT>
neg
</OUTPUT>
<INPUT>
i can only say that this was my favorite record of last year . this record is hard to find in your local record store , so search it out.. . incredible - ca n't wait to hear more !
</INPUT>
<OUTPUT>
pos 
</OUTPUT>
<INPUT>
i was looking for a pair of roof-prism binoculars like this , for hawk and bird watching . i had looked at swarovskis , but they were way out of my budget , nearly a thousand dollars for this resolution . these binoculars are crystal-clear , with sharp focus . the 42mm lens lets in lots of light , lots more than the 35 's i 've been using . i wear glasses and the twist - up - and -down eyecups are a blessing . there are also attached lens protectors for the 42mm , nice because i always used to lose them . for the price , these ca n't be beat
</INPUT>
<OUTPUT>
neg 
</OUTPUT>
</EXAMPLES>
"""


def parse_line(doc: str, task: str = "sentiment") -> Tuple[str, str]:
    data = doc.split(" ")
    label = data[0]
    sentiment = data[1]
    doc = " ".join(data[2:])

    if task == "sentiment":
        return (doc, sentiment)
    else:
        return (doc, label)


@dataclass
class PromptEvaluation:
    fp: int = 0
    fn: int = 0
    tp: int = 0
    tn: int = 0
    err: int = 0


def evaluate(y_pred: List[str], y_true: List[str]) -> PromptEvaluation:
    fp = 0
    fn = 0
    tp = 0
    tn = 0
    err = 0
    for y_hat, y in zip(y_pred, y_true):
        if y_hat not in ["pos", "neg"]:
            err += 1
        elif y_hat == "pos" and y == "pos":
            tp += 1
        elif y_hat == "neg" and y == "neg":
            tn += 1
        elif y_hat == "pos" and y == "neg":
            fp += 1
        elif y_hat == "neg" and y == "pos":
            fn += 1
    return PromptEvaluation(fp, fn, tp, tn, err)


def get_scores(eval: PromptEvaluation) -> Tuple[float, float, float, float]:
    rec = eval.tp / (eval.tp + eval.fp)
    prec = eval.tn / (eval.fn + eval.tn)
    err_rate = eval.err / (eval.tp + eval.tn + eval.fp + eval.fn + eval.err)
    return 2 * ((rec * prec) / (rec + prec)), rec, prec, err_rate


def predict(sample: str, client, prompt: str = prompt1_zero_shot) -> Tuple[str, str]:
    response = client.chat.completions.create(
        model="qwen-instruct",
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": f"{sample}"},
        ],
        temperature=0,
        max_tokens=10,
    )
    prediction = response.choices[0].message.content.strip().lower()
    return prediction, response


def main():
    with open("data/reviews-train.txt", "r") as fp:
        data = fp.readlines()
    XY_train = [parse_line(line) for line in tqdm(data)]

    client = OpenAI(base_url="http://129.213.94.57:8000/v1", api_key="dummy")
    ctr = 0
    for prompt in [
        prompt1_zero_shot,
        prompt2_one_shot,
        prompt3_xml_few_shot,
        prompt4_xml_few_shot,
    ]:

        y_pred = []
        y_true = []
        for x, y in tqdm(XY_train):
            y_hat, _ = predict(x, client, prompt)
            y_pred.append(y_hat)
            y_true.append(y)

        eval = evaluate(y_pred, y_true)
        f1, rec, prec, err_rate = get_scores(eval)
        print()
        print(f"{ctr}")
        print()
        print(f"err rate: {err_rate}")
        print(f"f1: {f1}")
        print(f"precision: {prec}")
        print(f"recall: {rec}")
        print(eval)
        ctr += 1


if __name__ == "__main__":
    main()
