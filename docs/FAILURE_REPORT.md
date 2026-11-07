================================================================================
                ICD-11 PIPELINE FAILURE & DIAGNOSTICS REPORT
================================================================================
Notes Evaluated:    60
Top-1 Accuracy:     50/60 (83.33%)
Top-3 Accuracy:     53/60 (88.33%)
Official Grader:    63.75 / 75.0  (Normalized: 85.00 / 100)
--------------------------------------------------------------------------------
CANDIDATE RECALL ON GOLD TARGETS:
  Recall@10:        53/60 (88.33%)
  Recall@30:        58/60 (96.67%)
  Recall@50:        60/60 (100.00%)
--------------------------------------------------------------------------------
FAILURE CATEGORIZATION TRIAGE (Non-Top-1 Notes):
  Cat 2 (Reranker Demotion)      :  6 notes ( 10.0%)
  Cat 3 (Specificity Error)      :  4 notes (  6.7%)
--------------------------------------------------------------------------------
PERFORMANCE BY CLINICAL STYLE:
  ambient_transcript     : n=15 | Top-1: 14/15 ( 93.3%) | Top-3: 14/15 ( 93.3%)
  short_consult          : n=16 | Top-1: 13/16 ( 81.2%) | Top-3: 15/16 ( 93.8%)
  typed_consult          : n=29 | Top-1: 23/29 ( 79.3%) | Top-3: 24/29 ( 82.8%)
--------------------------------------------------------------------------------
PERFORMANCE BY ICD-11 CHAPTER:
  Chapter 01             : n=13 | Top-1: 11/13 ( 84.6%) | Top-3: 11/13 ( 84.6%)
  Chapter 04             : n= 1 | Top-1:  0/ 1 (  0.0%) | Top-3:  0/ 1 (  0.0%)
  Chapter 05             : n= 7 | Top-1:  4/ 7 ( 57.1%) | Top-3:  4/ 7 ( 57.1%)
  Chapter 11             : n= 6 | Top-1:  5/ 6 ( 83.3%) | Top-3:  6/ 6 (100.0%)
  Chapter 12             : n= 7 | Top-1:  7/ 7 (100.0%) | Top-3:  7/ 7 (100.0%)
  Chapter 13             : n= 6 | Top-1:  5/ 6 ( 83.3%) | Top-3:  6/ 6 (100.0%)
  Chapter 14             : n= 3 | Top-1:  3/ 3 (100.0%) | Top-3:  3/ 3 (100.0%)
  Chapter 15             : n= 4 | Top-1:  4/ 4 (100.0%) | Top-3:  4/ 4 (100.0%)
  Chapter 16             : n= 7 | Top-1:  6/ 7 ( 85.7%) | Top-3:  6/ 7 ( 85.7%)
  Chapter 18             : n= 5 | Top-1:  4/ 5 ( 80.0%) | Top-3:  5/ 5 (100.0%)
  Chapter 24             : n= 1 | Top-1:  1/ 1 (100.0%) | Top-3:  1/ 1 (100.0%)
================================================================================
DETAILED PER-NOTE FAILURE TRIAGE:
Note ID    | Style            | Gold         | Predicted    | Category                 | Reason
--------------------------------------------------------------------------------------------------------------
dev-005    | short_consult    | 1F23.10      | 1F23.Z,1F23.Y,1F23.1Z | Cat 3 (Specificity Error) | Chose unspecified (.Z) 1F23.Z when specific 1F23.10 was gold
dev-009    | typed_consult    | 1B10.0       | 1B13.0,1B13.Z,1C62.2 | Cat 3 (Specificity Error) | Chose wrong specific sibling 1B13.0 instead of 1B10.0
dev-021    | typed_consult    | DA42.Z       | DA42.1,DA42.Z,DA42.41 | Cat 3 (Specificity Error) | Chose specific DA42.1 when unspecified (.Z) DA42.Z was gold
dev-022    | typed_consult    | GC08.Z       | JB40.3,JB40.Z,JA62.Z | Cat 2 (Reranker Demotion) | Gold GC08.Z was rank 5 in retrieval, but reranker placed it outside top 3
dev-042    | short_consult    | BA00.0       | BA00.Z,BA00.0,BB01.Z | Cat 3 (Specificity Error) | Chose unspecified (.Z) BA00.Z when specific BA00.0 was gold
dev-044    | typed_consult    | 5A21.0       | 5A11,5A40.0,5A41 | Cat 2 (Reranker Demotion) | Gold 5A21.0 was rank 12 in retrieval, but reranker placed it outside top 3
dev-046    | short_consult    | JB64.0       | JA64,JA63.2,JB64.0 | Cat 2 (Reranker Demotion) | Gold JB64.0 was rank 15 in retrieval, but reranker placed it outside top 3
dev-047    | ambient_transcript | 5A02.0       | 9B77,DA95,1D4B | Cat 2 (Reranker Demotion) | Gold 5A02.0 was rank 3 in retrieval, but reranker placed it outside top 3
dev-049    | typed_consult    | 4A84.0       | EK10.1,EK10.Z,4A84.Z | Cat 2 (Reranker Demotion) | Gold 4A84.0 was rank 9 in retrieval, but reranker placed it outside top 3
dev-057    | typed_consult    | 5A00.2Z      | FA23,5A00.Z,5A61.40 | Cat 2 (Reranker Demotion) | Gold 5A00.2Z was rank 2 in retrieval, but reranker placed it outside top 3
================================================================================
