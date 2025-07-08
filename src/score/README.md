## Scoring criteria for different types of short-answer Q&As

- **Forward (Full):** Exact match  
- **Forward (Phase names):** Jaccard similarity:

  \[
  \text{Jaccard} =
  \begin{cases}
  \frac{|GeneratedPhase \cap ExpectedPhases|}{|GeneratedPhase \cup ExpectedPhases|}, & \text{if } |GeneratedPhase \cup ExpectedPhases| > 0 \\
  0, & \text{otherwise}
  \end{cases}
  \]

- **Reverse:** Each generated answer is compared with all existing expected answers of the same element pair \( \{E_1, E_2, \dots, E_n\} \), and the final score is the maximum of the scoring function across all of them:

  \[
  \max_{E_i \in \text{Expected}} \left( 
    \frac{1}{2N} \sum_{l=1}^{N} \text{FractionAcc}_l(E_i) + 
    \frac{1}{2} \text{TempAcc}(E_i)
  \right)
  \]

  where:

  \[
  \text{FractionAcc}_l(E_i) = 1 - \left| \frac{\text{Expected\%}_l - \text{Generated\%}_l(E_i)}{100\%} \right|
  \]

  \[
  \text{TempAcc}(E_i) = 1 - \left| \frac{\text{Expected}_{K} - \text{Generated}_{K}(E_i)}{\text{maxK} - \text{minK}} \right|
  \]

  \[
  N = \text{number of elements in the target system}
  \]
