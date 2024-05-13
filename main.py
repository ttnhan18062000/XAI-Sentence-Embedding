from explainer import get_shap_values, plot_contributions, get_word_contributions
from utils import tokenize, _tokenize_sent

# s1 = "Vietnam's climate is diverse due to its varied topography and long latitude span. The country experiences a monsoon-influenced climate typical of mainland Southeast Asia"
# s2 = "Vietnam's climate is diverse due to its varied topography and long latitude span. The country experiences a monsoon-influenced climate typical of mainland Southeast Asia"
s1 = "What field does machine learning belong to?".lower()
# s2 = "Machine learning (ML) is a field of study in artificial intelligence concerned with the development and study of statistical algorithms that can learn from data and generalize to unseen data, and thus perform tasks without explicit instructions.[1] Recently, artificial neural networks have been able to surpass many previous approaches in performance."
s2 = """
Since deep learning and machine learning tend to be used interchangeably, it’s worth noting the nuances between the two. Machine learning, deep learning, and neural networks are all sub-fields of artificial intelligence. However, neural networks is actually a sub-field of machine learning, and deep learning is a sub-field of neural networks. The way in which deep learning and machine learning differ is in how each algorithm learns. "Deep" machine learning can use labeled datasets, also known as supervised learning, to inform its algorithm, but it doesn’t necessarily require a labeled dataset. The deep learning process can ingest unstructured data in its raw form (e.g., text or images), and it can automatically determine the set of features which distinguish different categories of data from one another. This eliminates some of the human intervention required and enables the use of large amounts of data. You can think of deep learning as "scalable machine learning" as Lex Fridman notes in this MIT lecture (link resides outside ibm.com).
""".lower()
specified_words = ["machine learning"]
multi_word_tokens = ["machine learning", "deep learning", "neural networks", "supervised learning", "artificial intelligence"]
# shap_values = get_shap_values(
#     s1, s2, n_samples=10, metric="euclids", vis=False, specified_words=None, multi_word_tokens=None
# )
shap_values = get_word_contributions(s1, s2, metric="euclid")
plot_contributions(shap_values)
