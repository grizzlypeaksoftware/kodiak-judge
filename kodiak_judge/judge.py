"""Which of two answers is better? Accept when confident, escalate when unsure (after JEV-as-a-Judge, arXiv:2609.26550).

Position bias: a judge can prefer whichever answer comes first (or second). On MT-Bench, swapping the two answers changed 18% of
Kodiak-v0.4-1B's verdicts (release audit, D69). Three defences, all on by default:
  1. Judge both orders and average the probabilities, so position effects cancel in the score.
  2. Report both orders' verdicts, so you can see when the order mattered.
  3. Escalate whenever the two orders disagree, however confident the average looks.
"""
from __future__ import annotations

from dataclasses import dataclass

QUESTION = [{"type": "choice", "id": "decision", "text": "Which answer is better?", "allow_null": False,
             "labels": [{"id": "first", "text": "the first answer"}, {"id": "second", "text": "the second answer"},
                        {"id": "tie", "text": "they are about equally good"}]}]
MIRROR = {"first": "second", "second": "first", "tie": "tie"}  # a verdict seen from the swapped order


@dataclass
class Judge:
    kodiak: object  # kodiak_s1.hub.Kodiak or KodiakEnsemble
    threshold: float = 0.7
    both_orders: bool = True
    escalate_on_disagreement: bool = True

    @classmethod
    def load(cls, model: str = "cortex-agent-llc/kodiak-v0.4-1b", **kw) -> "Judge":
        from kodiak_s1.hub import Kodiak
        return cls(Kodiak.from_pretrained(model), **kw)

    def judge(self, question: str, answer_1: str, answer_2: str, **kw) -> dict:
        return self.judge_many([{"question": question, "answer_1": answer_1, "answer_2": answer_2}], **kw)[0]

    def judge_many(self, items: list[dict], threshold: float | None = None) -> list[dict]:
        """items: [{"question", "answer_1", "answer_2"}] -> one verdict per item (labels: first / second / tie)."""
        t = self.threshold if threshold is None else threshold
        reqs = []
        for it in items:
            st = {"question": it["question"], "answer_1": it["answer_1"], "answer_2": it["answer_2"]}
            reqs.append({"state": st, "questions": QUESTION})
            if self.both_orders:
                reqs.append({"state": {**st, "answer_1": it["answer_2"], "answer_2": it["answer_1"]}, "questions": QUESTION})
        outs = self.kodiak.answer(reqs)
        step, res = (2 if self.both_orders else 1), []
        for i in range(len(items)):
            p1 = outs[i * step]["answers"]["decision"]["probs"]
            as_given = max(p1, key=p1.get)
            if self.both_orders:
                p2 = outs[i * step + 1]["answers"]["decision"]["probs"]
                swapped = {k: p2[MIRROR[k]] for k in p1}  # the swapped run, translated back to the original order
                p = {k: (p1[k] + swapped[k]) / 2 for k in p1}
                when_swapped = max(swapped, key=swapped.get)
                disagree = as_given != when_swapped
            else:
                p, when_swapped, disagree = p1, None, None
            verdict = max(p, key=p.get)
            if p[verdict] < t:
                action, reason = "escalate", f"confidence {p[verdict]:.2f} is below the threshold {t:.2f}"
            elif disagree and self.escalate_on_disagreement:
                action, reason = "escalate", "the verdict changed when the answers were swapped (position bias)"
            else:
                action, reason = "accept", "confident, and the same verdict in both orders" if self.both_orders else "confident"
            res.append({"verdict": verdict, "confidence": round(p[verdict], 4), "probs": {k: round(v, 4) for k, v in p.items()},
                        "action": action, "reason": reason, "threshold": t,
                        "orders": {"as_given": {"verdict": as_given, "probs": {k: round(v, 4) for k, v in p1.items()}},
                                   "swapped": ({"verdict": when_swapped, "probs": {k: round(v, 4) for k, v in swapped.items()}}
                                               if self.both_orders else None),
                                   "disagree": disagree}})
        return res
