from explainer import get_word_contributions
import pandas as pd
import argparse
from tqdm import tqdm
import pickle

def main(metric, input_path, output_path, batch_size):
    data = pd.read_csv(input_path)
    documents = data["doc"].tolist()
    documents_batches = [documents[i : i + batch_size] for i in range(0, len(documents), batch_size)]
    for i, document_batch in enumerate(documents_batches): 
        words_importants = []
        for doc in tqdm(document_batch):
            words_important = get_word_contributions(doc, doc, metric=metric)
            words_importants.append(words_important)
        with open(f"{output_path}/{metric}_{i*batch_size}_{i*batch_size + batch_size}.pkl", 'wb') as f:
            pickle.dump(words_importants, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', dest='input_path', type=str)
    parser.add_argument('--output', dest='output_path', type=str)
    parser.add_argument('--batch-size', dest='batch_size', type=int)
    parser.add_argument('--metric', dest='metric', type=str)
    # Parse arguments
    args = parser.parse_args()
    main(args.metric, args.input_path, args.output_path, args.batch_size)