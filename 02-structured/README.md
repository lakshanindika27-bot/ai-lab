
## Retry behaviour (hard set, 10 questions, llama3.2:3b, temperature 0)

- 7 of 10 questions triggered retries. None of them was rescued by a retry.
- In 5 of the 7, the second reply was identical to the first (the loop stopped after 2 calls);
  in the other 2 the first two replies differed, and a later reply repeated an earlier one.
- Stopping as soon as a reply repeats keeps the eval result unchanged
  (3/4 answered, 3/4 cited, 6/6 abstained, before and after) and cut total time from 241s to 162s.
- Caveat: tiny sample, and run-to-run timing varies (an earlier run of the same code took 225s).
