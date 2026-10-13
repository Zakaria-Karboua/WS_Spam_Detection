# Spam Detection with a Dense Neural Network in PyTorch: Report

*Workshop module · ENSM · Unit 1*

## 1. Experimental protocol

**Environment.** See `results/run_config.json` (Python, PyTorch, device, platform). *Copy the values here.*

**Dataset.** Kaggle, *Spam Email Detection Dataset (Clean & ML Ready)*: 10,000 emails, 20 columns, 6,005 ham (60.05 %) and 3,995 spam (39.95 %). Only `email_text` is vectorized. The columns `subject` and `sender_domain` are excluded because each of their 8 values occurs in only one class. The messages appear to be synthetically generated (spam begins with a run of spam keywords followed by random filler words). *Add the dataset's Kaggle link and licence.*

**Balancing.** The majority class (ham) is undersampled to the size of the minority class (3,995), giving 7,990 messages (50 / 50). The balanced set is then split per class into a training set (3,196 + 3,196 = 6,392) and a test set (799 + 799 = 1,598), so both sets are balanced.

**Vectorization.** Tokenization: lowercasing, punctuation and whitespace removed, tokens = words, numbers, URLs and the symbols £ $ € %. The vocabulary (the N = 500 tokens present in the most training messages, minimum 2 messages) and the IDF values are computed on the training set only. Three representations: One-Hot (presence 0/1), Bag of Words (counts), TF-IDF (`TF × log(N/DF)` with `TF` = count / message length).

**Network.** Input 500 → 41 hidden layers × 64 neurons (Linear + ReLU, built in a loop) → 1 output (raw score). He (Kaiming normal) initialization, zero biases. Loss: Binary Cross-Entropy on logits. Optimizer: Adam, lr = *(from run_config.json)*, 1000 epochs, full batch, seed 42.

| Hyperparameter | Value |
|---|---|
| Hidden layers L | 41 |
| Neurons per layer | 64 |
| Vector size N | 500 |
| Epochs | 1000 |
| Learning rate | *fill in* |
| Seed | 42 |

## 2. Comparative analysis: One-Hot vs BoW vs TF-IDF

*Paste the table from `results/metrics.md` (accuracy, precision, recall, F1, false-positive rate, confusion matrix) and `results/confusion_matrices.png`.*

*Discuss:*
* *Which representation gives the best test accuracy and the lowest false-positive rate? Are the differences larger than what one or two messages would explain (the test set has 1,598 messages)?*
* *Note that a logistic regression separates this dataset perfectly, so accuracy alone cannot rank the representations. Consider repeating the study on the SMS Spam Collection.*

## 3. Convergence study

*Insert `results/loss_curves.png` and `results/gradient_flow.png`.*

*Discuss:*
* *How quickly does each representation leave the initial loss (about 0.69)? Is any curve unstable, flat, or divergent (`diverged` column in `metrics.md`)?*
* *Input scale: One-Hot values are 0/1, BoW counts reach 7, TF-IDF values stay near or below 1.2. Link this to the first-layer activations and the early loss.*
* *Gradient flow: compare the gradient norm of the first layers with the last layers at epochs 1, 500 and 1000. Is there vanishing or exploding behaviour in the 41-layer plain network?*
* *Weight initialization: He initialization keeps the variance of the activations roughly constant through ReLU layers (Var(w) = 2 / fan_in). Run `--compare-init` and compare with PyTorch's default initialization (smaller weight variance, which shrinks signals layer after layer in a deep ReLU network).*

## 4. Conclusion

*Summarize what the experiments show about end-to-end learning versus manual feature extraction: the network was given raw token vectors and no hand-crafted features (word frequencies of chosen words, capital-letter ratio, link counts, message length). State what it learned on its own, and be explicit about the limits of this dataset (synthetic, perfectly separable, 60/40 imbalance) and of the evidence.*
