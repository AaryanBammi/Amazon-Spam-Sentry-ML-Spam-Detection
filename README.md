# Amazon Review Spam Detection

Binary classifier that flags spam reviews in Amazon's Clothing, Shoes & Jewelry category, using engineered text-quality and sentiment features alongside TF-IDF.

**Stack:** Python, scikit-learn, NLTK, VADER, textstat, mlxtend  
**Context:** Team project, BU Questrom.

## Problem
Fake and low-quality reviews erode customer trust and distort product rankings. The goal is a model that can screen reviews at scale before they influence buyers.

## Data
Amazon Clothing, Shoes & Jewelry reviews with a spam / not-spam label. Narrowed to the 5,000 most-reviewed products, then downsampled the majority class to balance spam and non-spam. The raw JSON file is not included in this repo.

## Approach
- **Custom sklearn transformers** for text cleanup (lemmatisation, stopwords), VADER sentiment, Flesch readability, review length, unique-word count and review-to-product similarity
- `ColumnTransformer` pipeline combining TF-IDF text features with scaled numeric and one-hot categorical features
- 70/30 train-test split; tuned SGD, Random Forest and SVM with RandomizedSearchCV and HalvingGridSearchCV (batched to fit memory)
- Combined the tuned models in a hard-voting ensemble

## Results (test set, about 135K reviews)
| Model | Accuracy | Macro F1 |
|---|---|---|
| SGD classifier | 0.84 | 0.84 |
| Random Forest | 0.82 | 0.82 |
| SVM | 0.82 | 0.82 |
| Voting ensemble | 0.84 | 0.84 |

Sentiment, product similarity and words such as "return", "love" and helpful-vote signals ranked as the strongest predictors.

## Next steps
- Try transformer embeddings for richer text signal
- Do error analysis on false positives, which carry the highest business cost
- Serve the model behind a simple API for batch scoring

## Repo contents
- `Amazon_Spam_Detection.ipynb`
- `requirements.txt`
