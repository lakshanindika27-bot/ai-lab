
## Retry behaviour (hard set, 10 questions, llama3.2:3b, temperature 0)

- 7 of 10 questions triggered retries. None of them was rescued by a retry.
- In 5 of the 7, the second reply was identical to the first (the loop stopped after 2 calls);
  in the other 2 the first two replies differed, and a later reply repeated an earlier one.
- Stopping as soon as a reply repeats keeps the eval result unchanged
  (3/4 answered, 3/4 cited, 6/6 abstained, before and after) and cut total time from 241s to 162s.
- Caveat: tiny sample, and run-to-run timing varies (an earlier run of the same code took 225s).
- Retry temperatures `0,0.3,0.7` (via `RETRY_TEMPS`) produced more varied replies
  (all-identical retries 7 -> 4) but still rescued 0 of 7 retried questions, and the eval
  result did not change (3/4 answered, 3/4 cited, 6/6 abstained). Default stays `0,0,0`.
  Same tiny sample; the run took longer (422s vs 162s), partly because early exit fires less often.
- Why retries do not help: 6 of the 7 retried questions were unanswerable ones where the model
  kept returning schema-valid replies with answerable=true and a real source name. The evidence
  check rejected them each time, so the final result was the correct abstention either way.
  The remaining retried question (image vs container, expected answerable) also got identical
  valid replies and was rejected by a later check; the script does not say which one.

## Few-shot abstention examples (FEWSHOT=1)

Two synthetic examples (one answerable, one abstain, unrelated topics) added to the prompt.
The eval sets were not used to write them.

| Set | Metric | FEWSHOT=0 | FEWSHOT=1 |
|---|---|---|---|
| hard (10) | answered / cited / abstained | 3/4, 3/4, 6/6 | 2/4, 2/4, 6/6 |
| heldout (19) | answered / cited / abstained | 10/10, 10/10, 9/9 | 9/10, 9/10, 8/9 |

No improvement: abstention did not rise and one answerable question more was missed on each set.
Differences are one question each, so this is a direction, not proof. Default stays off.
