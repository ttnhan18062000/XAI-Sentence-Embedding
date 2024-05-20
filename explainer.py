from nltk.tokenize import word_tokenize
import nltk
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import torch
from transformers import BertTokenizer, BertForMaskedLM
import copy
import random
from tqdm import tqdm
import plotly.express as px
import umap
import matplotlib.pyplot as plt

from utils import build_feature, encode, _tokenize_sent, generate_unique_random_subsets

nltk.download('punkt')

# Check and set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load pre-trained models
masked_model = BertForMaskedLM.from_pretrained("bert-large-cased").to(device)
tokenizer = BertTokenizer.from_pretrained("bert-large-cased")
w_tokenizer = word_tokenize

# Choose appropriate SentenceTransformer model based on device availability
model_name = "all-MiniLM-L6-v2"
if torch.cuda.is_available():
    model = SentenceTransformer(model_name, device="cuda")
else:
    model = SentenceTransformer(model_name)

generated_strings = []


def batch_replace_masks(sentences, max_k=5, excludes_list=None):
    # Tokenize sentences
    tokenized_sentences = [tokenizer.tokenize(sentence) for sentence in sentences]

    # Find mask indices
    mask_indices = [
        [i + 1 for i, v in enumerate(tokens) if v == "[MASK]"]
        for tokens in tokenized_sentences
    ]

    # Encode sentences and move them to device
    encoded_inputs = tokenizer.batch_encode_plus(
        sentences, padding=True, truncation=True, return_tensors="pt"
    )
    encoded_inputs = {key: val.to(device) for key, val in encoded_inputs.items()}

    # Get masked model predictions
    masked_outputs = masked_model(**encoded_inputs)

    # Initialize result lists
    results = []
    predicted_words_list = []
    predicted_weights_list = []

    # Iterate over mask indices and predictions
    for (
        mask_indices_batch,
        predictions_batch,
        tokenized_sentence,
        excludes_batch,
    ) in zip(mask_indices, masked_outputs[0], tokenized_sentences, excludes_list):
        # Sort predictions and calculate weights
        sorted_predictions, sorted_indices = predictions_batch.sort(
            dim=-1, descending=True
        )
        weights = sorted_predictions / sum(sorted_predictions)

        k = 0
        top_k = max_k
        while k < top_k and top_k < 20:
            predicted_tokens = [
                sorted_indices[mask_index, k].item()
                for mask_index in mask_indices_batch
            ]
            predicted_weights = [
                weights[mask_index, k].item() for mask_index in mask_indices_batch
            ]
            predicted_words = [
                tokenizer.convert_ids_to_tokens([predicted_token])[0]
                for predicted_token in predicted_tokens
            ]

            # Check for exclusion
            is_match = False
            if excludes_batch:
                predicted_words_lowercase = [word.lower() for word in predicted_words]
                excludes_lowercase = [word.lower() for word in excludes_batch]
                is_match = any(
                    predicted_word == exclude_word
                    for predicted_word, exclude_word in zip(
                        predicted_words_lowercase, excludes_lowercase
                    )
                )

            # If not excluded, construct new sentence
            if not excludes_batch or not is_match:
                new_sentence = ""
                word_index = 0
                for token in tokenized_sentence:
                    if token != "[MASK]":
                        new_sentence += token + " "
                    else:
                        new_sentence += predicted_words[word_index] + " "
                        word_index += 1
                results.append(new_sentence[:-1].replace(" ##", ""))
                predicted_words_list.append(predicted_words)
                predicted_weights_list.append(predicted_weights)
            else:
                top_k += 1
            k += 1
    return results, predicted_words_list, predicted_weights_list


def batch_mask_sentences(
    features_dict, feature_name, no_mask_features, n_max_masks, n_samples, main_mask_token, sub_mask_token
):
    # Initialize lists to store results
    exclude_sentences = []
    include_sentences = []
    exclude_ignores = []
    include_ignores = []

    # Generate list of masked sentences
    masked_features = [
        feature
        for feature in features_dict.keys()
        if feature != feature_name and feature not in no_mask_features
    ]
    random_masked_features_list = generate_unique_random_subsets(masked_features, n_samples, n_max_masks)

    # Iterate through each desired number of masks
    # for n_mask in n_masks:
    for random_masked_features in random_masked_features_list:  # Generate samples
        # Initialize lists for this sample
        exclude_sent = []
        include_sent = []
        exclude_ign = []
        include_ign = []

        # Process each feature in the input dictionary
        for feature, word in features_dict.items():
            if feature == feature_name:  # Keep the target feature intact
                include_sent.append(word)
                exclude_sent.append(main_mask_token)
            elif feature in random_masked_features:  # Mask selected features
                exclude_sent.append(sub_mask_token)
                exclude_ign.append(word)
                include_sent.append(sub_mask_token)
                include_ign.append(word)
            else:  # Keep other features intact
                exclude_sent.append(word)
                include_sent.append(word)

        # Append results for this sample to respective lists
        exclude_sentences.append(exclude_sent)
        include_sentences.append(include_sent)
        exclude_ignores.append(exclude_ign)
        include_ignores.append(include_ign)

    # Construct and return the result dictionary
    result = {
        "exclude_sent": exclude_sentences,
        "include_sent": include_sentences,
        "exclude_ign": exclude_ignores,
        "include_ign": include_ignores,
    }
    return result


def get_sim_scores(sent_pairs, ignores, s1len, metric="cosine"):
    # Generate lists for sentence pairs split by s1len
    s1_list = [" ".join(sent_pair[:s1len]) for sent_pair in sent_pairs]
    s2_list = [" ".join(sent_pair[s1len:]) for sent_pair in sent_pairs]

    # Divide lists into batches of size 32
    batch_size = 32
    s1_lists = [s1_list[i : i + batch_size] for i in range(0, len(s1_list), batch_size)]
    s2_lists = [s2_list[i : i + batch_size] for i in range(0, len(s2_list), batch_size)]

    # Replace masks in batches
    replaced_s1_list = []
    replaced_s2_list = []
    for s1l, s2l in zip(s1_lists, s2_lists):
        replaced_s1l, _, _ = batch_replace_masks(s1l, max_k=1, excludes_list=ignores)
        replaced_s2l, _, _ = batch_replace_masks(s2l, max_k=1, excludes_list=ignores)
        replaced_s1_list.extend(replaced_s1l)
        replaced_s2_list.extend(replaced_s2l)
    generated_strings.extend(replaced_s2_list)

    # Generate masked model inputs
    masked_model_inputs = [
        [r_s1, r_s2] for r_s1, r_s2 in zip(replaced_s1_list, replaced_s2_list)
    ]

    # Encode sentences to get scores and embeddings
    scores, s1_embeds, s2_embeds = encode(model, masked_model_inputs, metric=metric)

    # Create dictionary for embeddings
    embeds_dict = {
        "s1_embeds": s1_embeds,
        "s2_embeds": s2_embeds,
        "s1_list": replaced_s1_list,
        "s2_list": replaced_s2_list,
    }

    return scores, embeds_dict


def get_shap_value(
    features_dict, f_name, no_mask_features, n_max_masks, n_samples, s1len, main_mask_token, sub_mask_token, metric="cosine", vis=False
):
    # Batch mask sentences
    masked_sentences = batch_mask_sentences(
        features_dict=features_dict,
        feature_name=f_name,
        no_mask_features=no_mask_features,
        n_max_masks=n_max_masks,
        n_samples=n_samples,
        main_mask_token=main_mask_token,
        sub_mask_token=sub_mask_token,
    )

    # Get similarity scores for both include and exclude sentences
    exclude_scores, ex_embeds_dict = get_sim_scores(
        sent_pairs=masked_sentences["exclude_sent"],
        ignores=masked_sentences["exclude_ign"],
        s1len=s1len,
        metric=metric,
    )
    include_scores, in_embeds_dict = get_sim_scores(
        sent_pairs=masked_sentences["include_sent"],
        ignores=masked_sentences["include_ign"],
        s1len=s1len,
        metric=metric,
    )

    # Calculate average scores
    avg_scores = [
        i_score - e_score for i_score, e_score in zip(include_scores, exclude_scores)
    ]
    shap_value = sum(avg_scores) / len(avg_scores)

    return shap_value, ex_embeds_dict["s2_embeds"], in_embeds_dict["s2_embeds"]


def get_shap_values(s1, s2, n_samples, main_mask_token, sub_mask_token, metric="cosine", vis=False, specified_words=None, multi_word_tokens=None):
    """
    s1: first sentence
    s2: second sentence, the sentence will be calculated
    n_samples: number of masked sentence generated
    main_mask_token: token used for replacing the main feature when calculating contribution, [MASK] will be replaced using masked language models
    sub_mask_token: token used for replacing the randomize masked features when calculating contribution, [MASK] will be replaced using masked language models
    metric: "cosine", "euclid", metric for evaluating similarity
    vis: visualize the contributions
    specified_words: Some target specific features to evaluate contributions; leave as None to evaluate the entire s2
    multi_word_tokens: grouped features as single token
    """
    # Build feature dictionary and get lengths
    df_features, s1len, s2len = build_feature(s1, s2, multi_word_tokens=multi_word_tokens)
    features_dict = df_features.to_dict()
    features_dict = {k: v[0] for k, v in features_dict.items()}

    # Determine number of features and maximum masks
    n_max_masks = int(s2len * 0.5)
    no_mask_features = list(features_dict.keys())[:s1len]

    # Determine feature names
    if specified_words:
        f_names = [
            k for k, v in features_dict.items() if v in specified_words and "s2" in k
        ]
    else:
        f_names = list(features_dict.keys())[s1len:]

    shap_values = {}
    sen_embeddings = []
    labels = []
    texts = []

    # Calculate SHAP values for each feature
    for idx, f_name in tqdm(enumerate(f_names)):
        shap_value, ex_embeds, in_embeds = get_shap_value(
            features_dict,
            f_name,
            no_mask_features=no_mask_features,
            n_max_masks=n_max_masks,
            n_samples=n_samples,
            s1len=s1len,
            main_mask_token=main_mask_token, 
            sub_mask_token=sub_mask_token,
            metric=metric,
            vis=False,
        )
        shap_values[f"{f_name}_{features_dict[f_name]}"] = shap_value
        print(f"{features_dict[f_name]} {shap_value}")
        if vis:
            sen_embeddings.extend(ex_embeds)
            sen_embeddings.extend(in_embeds)
            labels.extend([idx for i in range(len(ex_embeds) * 2)])
            texts.extend(
                [f"{features_dict[f_name]}_Exclude" for i in range(len(ex_embeds))]
            )
            texts.extend(
                [f"{features_dict[f_name]}_Include" for i in range(len(ex_embeds))]
            )

    if vis:
        visualize_embeddings(sen_embeddings, labels, texts)
    with open("generated_strings1.txt", "w",  encoding="utf-8") as f:
        for generated_string in generated_strings:
            f.writelines(generated_string + "\n")
    return shap_values


def get_word_contributions(sentence1, sentence2, metric="cosine"):
    sentence1_copy = copy.deepcopy(sentence1)
    sentence2_copy = copy.deepcopy(sentence2)

    df_features, s1_len, s2_len = build_feature(sentence1_copy, sentence2_copy)
    features_dict = df_features.to_dict()
    features_dict = {k: v[0] for k, v in features_dict.items()}
    num_features = len(features_dict)

    feature_names = list(features_dict.keys())[s1_len:]
    contributions = {}
    includes = []
    excludes = []
    for feature_name in feature_names:
        exclude_r_s1 = " ".join(list(features_dict.values())[:s1_len])
        exclude_r_s2 = " ".join(
            [w for k, w in list(features_dict.items())[s1_len:] if k != feature_name]
        )
        excludes.append([exclude_r_s1, exclude_r_s2])
        include_r_s1 = " ".join(list(features_dict.values())[:s1_len])
        include_r_s2 = " ".join(list(features_dict.values())[s1_len:])
        includes.append([include_r_s1, include_r_s2])
    include_scores, s1_embeds, s2_embeds = encode(model, includes, metric=metric)
    exclude_scores, s1_embeds, s2_embeds = encode(model, excludes, metric=metric)
    differences = [
        include_score - exclude_score
        for include_score, exclude_score in zip(include_scores, exclude_scores)
    ]
    return {
        f"{feature_name}_{features_dict[feature_name]}": diff
        for feature_name, diff in zip(feature_names, differences)
    }


def plot_contributions(shap_values):
    plt.figure(figsize=(15, 5))
    labels = [k.rsplit("_", 1)[1] for k in shap_values.keys()]
    bars = plt.bar(range(len(shap_values)), shap_values.values())
    for i, bar in enumerate(bars):
        y_val = labels[i]
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            y_val,
            ha="center",
            va="bottom",
        )
    plt.xticks(rotation=45)
    plt.show()


def visualize_embeddings(sentence_embeddings, labels, texts):
    reducer = umap.UMAP()
    embeddings = reducer.fit_transform(sentence_embeddings)
    df_embeddings = pd.DataFrame(embeddings, columns=["x", "y"])
    df_embeddings["label"] = labels
    df_embeddings["text"] = texts

    fig = px.scatter(
        df_embeddings,
        x="x",
        y="y",
        color="label",
        labels={"color": "label"},
        hover_data=["text"],
        title="Embedding Visualization",
    )
    fig.show()
    fig.write_html("plot.html")
