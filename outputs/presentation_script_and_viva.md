# Presentation script

## Slide 1: Research Collaborator Recommendation
Social Network Analysis AAT A structural link prediction experiment on Stanford SNAP ca-GrQc

## Slide 2: Problem and objectives
Rank currently unconnected researchers using co-authorship structure. Compare four transparent scores and evaluate missing-link recovery. Export exact global top 10 and top 20 rankings.

## Slide 3: Dataset and graph model
5,242 researcher IDs and 14,484 cleaned co-authorship edges. Simple, undirected and unweighted graph. arXiv GR-QC papers: January 1993 to April 2003. Source: snap.stanford.edu/data/ca-GrQc.html

## Slide 4: Four prediction methods
Common Neighbors: number of shared collaborators. Jaccard: shared / combined collaborators. Adamic-Adar: sum of 1 / ln(shared neighbor degree). Preferential Attachment: degree(u) x degree(v).

## Slide 5: Reproducible evaluation
Primary seed 42. Random 80/20 split of unique edges. Keep every node in training, including isolates. Use one full-graph nonedge per held-out positive. Score only the training graph. Ties use ascending IDs.

## Slide 6: Evaluation metrics
ROC-AUC measures positive/negative ranking quality. Precision@K = hits / K. Recall@K = hits / test positives. F1@K is the harmonic mean of precision and recall at K. Balanced sampled precision differs from all-nonedge precision.

## Slide 7: Primary results
Highest primary ROC-AUC: Adamic-Adar (0.88894). Winner Precision@10: 1.000; Precision@20: 1.000. Same test results select the winner, without independent validation.

## Slide 8: Split sensitivity
Repeat the protocol with seeds 7, 21 and 84. The primary split remains seed 42. The exported sensitivity tables report all metrics and mean/SD.

## Slide 9: Exact collaborator recommendations
All 13,722,177 unordered nonedges are scored. Four heaps retain only 20 records each, with no candidate filter. Example (Adamic-Adar): IDs 45 and 46, 42 shared collaborators. Final recommendations use the full observed graph.

## Slide 10: Network visualizations
Bounded subset. Gray = observed; orange dashed = predicted. Full-network charts, offline HTML and Cytoscape exports accompany this view.

## Slide 11: Application demonstration
Open Overview, then select an ID in Network Explorer. Compare metrics and ROC curves in Algorithm Comparison. Switch method and top 10/20, explain a pair, download CSV. Open Researcher Profile for local structure and personal suggestions.

## Slide 12: Limitations and conclusion
IDs supply no names, interests or compatibility evidence. Sampled nonedges are assumed negatives. Random edge recovery does not prove future prediction. Structural scores provide explainable possibilities for further study.

## Live app sequence
1. Run `python -m streamlit run app.py`.
2. Read the graph counts on Overview.
3. In Network Explorer select a researcher and show predicted links. Explain the subset label.
4. Open Algorithm Comparison and distinguish training-only scoring from full-network recommendations.
5. In Collaborator Recommendations switch all four algorithms and top 10/20, explain a selected rank and download CSV.
6. Open Researcher Profile and inspect shared collaborator IDs.
7. Finish with About and Limitations.

# Viva questions

1. **What is a node?** An anonymized researcher represented by the supplied integer ID.

2. **What is an edge?** An observed co-authorship relation between two researchers.

3. **Why undirected?** Co-authorship is mutual.

4. **Why remove self-loops?** Collaborating with oneself is not a distinct collaborator relation.

5. **Why remove duplicates?** The model records existence of collaboration, not its frequency.

6. **What does degree mean?** The number of distinct observed collaborators.

7. **What is clustering?** The fraction of possible links among a node's neighbors that exist, averaged over all nodes, with zero for degree below two.

8. **How does Common Neighbors work?** It counts researchers connected to both endpoints.

9. **Why use Jaccard?** It normalizes common neighbors by the union of neighbor sets.

10. **Why Adamic-Adar?** It downweights very popular shared collaborators using inverse log degree.

11. **What does Preferential Attachment favor?** Pairs whose endpoint degrees have a large product.

12. **How is leakage prevented?** All evaluation scores use training edges only; held-out edges supply labels.

13. **Why sample negatives from the full graph complement?** To avoid labeling held-out true edges as negatives.

14. **Are negatives certain?** No. Missing observed edges are assumed negatives for evaluation.

15. **What does AUC mean?** The probability a positive ranks above a negative, with half credit for ties.

16. **What is Recall@20?** Held-out positives found among the first 20 ranked evaluation pairs divided by all held-out positives.

17. **Why is recall small?** Only K slots can retrieve thousands of test positives.

18. **What is F1@K?** The harmonic mean of precision and recall at that K.

19. **How are ties resolved?** Descending score, then ascending canonical endpoint IDs.

20. **Does the best method have independent validation?** No. It is chosen descriptively using the same primary test results.

21. **Do these results prove future prediction?** No. A random edge split evaluates missing-link recovery, not chronological forecasting.

22. **Are global rankings approximate?** No. Every unordered full-graph nonedge is scored, and small heaps retain the exact top 20.

23. **Why keep all training nodes?** Removing isolated nodes would change the candidate population and exclude difficult examples.

24. **Why use separate full-network rankings?** After evaluation, all observed structure is available for current recommendations.

25. **How do the app and notebook agree?** They import the same score and ranking functions and read the same exported results.

