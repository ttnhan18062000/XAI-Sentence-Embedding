from explainer import get_shap_values, plot_contributions, get_word_contributions
from utils import tokenize, _tokenize_sent

s1 = "the method is more effiecent than naive methods ."
s2 = "the methodology takes much less time rather than naive methods ."
specified_words = []
multi_word_tokens = []
shap_values = get_shap_values(
    s1, s2, n_samples=50, metric="cosine", main_mask_token="", sub_mask_token="", vis=False, specified_words=None, multi_word_tokens=None
)
# shap_values = get_word_contributions(s1, s2, metric="euclid")
plot_contributions(shap_values)
