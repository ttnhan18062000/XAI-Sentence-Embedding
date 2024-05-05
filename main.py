from explainer import get_shap_values, plot_contributions
from utils import tokenize

# s1 = "Vietnam's climate is diverse due to its varied topography and long latitude span. The country experiences a monsoon-influenced climate typical of mainland Southeast Asia"
# s2 = "Vietnam's climate is diverse due to its varied topography and long latitude span. The country experiences a monsoon-influenced climate typical of mainland Southeast Asia"
s1 = "What is Machine Learning?"
s2 = "Machine learning (ML) is a field of study in artificial intelligence concerned with the development and study of statistical algorithms that can learn from data and generalize to unseen data, and thus perform tasks without explicit instructions.[1] Recently, artificial neural networks have been able to surpass many previous approaches in performance."
specified_words = tokenize(
    "Machine learning (ML) a field artificial intelligence concerned of statistical artificial neural networks have been approaches"
)
shap_values = get_shap_values(
    s1, s2, n_samples=5, vis=False, specified_words=specified_words
)
print(sum(list(shap_values.values())))
plot_contributions(shap_values)
