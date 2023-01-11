# Paper peer review

## Diaz ─ Combining EA and MCTS for improving optimization algorithms

- 1.1
  - Maybe defining ML wouldn't be necessary for a publication in a ML journal?
  - The explanation of the 4 steps is very clear.
- 1.2
  - It was a good idea to link the phenotype in biology to what we actually mean in the computational sense
  - Figure 2 helps understanding and illustrating, nice.
- 2.1
  - The explanation of concepts is well done and clear (_e.g._ the Swiss tournament)
- 3.1
  - Write numbers in letters
- Size
  - The article is 9 pages. It was supposed to be 5-8 pages, including sources.
- I believe that citations should be at the end of a sentence, as an addition, not making it the subject of the sentence. E.g. "Some methods implement this awesome feature [4].", not like "Method [4] implements this awesome feature."

## Birbiri ─

- Grammar mistakes.
- Article length is 9 pages. Maximum allowed is 8 pages, including title and citations. Interline space & margins could be reduced to fit. everything into 8 pages.
- Some titles are in french.
- Use LaTeX built-in counters, otherwise you end up with mistakes such as having two subsubsections "5", like at the end of page 1.
- I think it would be nice to introduce your subsubsections with at least a short sentence, rather that putting them straight after the subsection title like 1. and 1.1.
- Some citations are empty, like [6] and [14]
- Some citations point to library documentation. Do we really need to know each function you used? I would have assumed you would describe the techniques implemented by some papers, not explain that you convert Numpy arrays with a `from_numpy_matrix()` function. This is not the State of the Art.
- I don't know what the "f" in "fMRI" stands for. I can assume a reader is expected to know that, but what about the "COO format"?
- lambda should be written in maths mode (Greek symbol), not in Latin alphabet.
- Section 4 contains a missing reference error ([?]).
- You implemented an optimizer, you chose a learning rate and you selected a model. Saying "The possible reasons for this problem could be the wrong optimizer, learning rate, or wrong model selection." basically means that anything could have gone wrong and you have not investigated. If you have investigated, you did not mention it.
- You say that you are doing a "literature search", but that you were supposed to study the literature prior to this report.

Overall comment: I think that you missed what is expected for the State of the Art (SOTA) review. You were supposed to study what the best techniques available out there are for this specific field, choose a few and explain the concepts behind them with a high-level view. I don't believe the SOTA should contain your work. I don't believe it should include every single implementation detail from your code, because these are details. I don't believe it should include the type of the variables you use because the SOTA should be a higher level view, explaining the concepts behind a few of the best techniques.
