

from ragas.dataset_schema import SingleTurnSample
from ragas.metrics import BleuScore,RougeScore,ExactMatch,StringPresence

# Wrap your sample
sample = SingleTurnSample(
    user_input="translate to French: Hello, world!",
    response="Bonjour, le monde!",
    reference="Bonjour le monde!"
)

metric = BleuScore()  # non‑LLM n‑gram precision + brevity penalty :contentReference[oaicite:1]{index=1}
result = metric.single_turn_score(sample)
print("BLEU:", result)  # e.g. 0.840 …



sample = SingleTurnSample(
    user_input="summarize: …",
    response="Key points are A, B, and C.",
    reference="The summary should cover A, B, and C."
)

metric = RougeScore(rouge_type="rougeL")  # LCS‑based recall/precision/F1 :contentReference[oaicite:2]{index=2}
result = metric.single_turn_score(sample)
print("ROUGE‑L:", result)  # e.g. 0.67

sample = SingleTurnSample(
    user_input="What is 2 + 2?",
    response="4",
    reference="4"
)

metric = ExactMatch()  # 1.0 if identical, else 0.0 :contentReference[oaicite:3]{index=3}
result = metric.single_turn_score(sample)
print("ExactMatch:", result) 




# sample = SingleTurnSample(
#     user_input="paraphrase: The cat sat on the mat.",
#     response="A cat was sitting on the rug.",
#     reference="The cat was seated on the rug."
# )

# metric = BertScore(model_name="microsoft/deberta-xlarge-mnli")  # embedding‑based semantic sim :contentReference[oaicite:5]{index=5}
# result = metric.single_turn_score(sample)
# print("BERTScore F1:", result.value)  