# Data

Place the Kaggle CSV here as `data/spam_email_dataset.csv`
("Spam Email Detection Dataset (Clean & ML Ready)" on Kaggle).
The file is git-ignored so the repository does not redistribute the dataset.

Columns used by default: `email_text` (the raw message) and `label` (0 = ham, 1 = spam).

Another dataset can be used by passing its columns, e.g. the SMS Spam Collection:

```bash
python run_experiments.py --data data/spam.csv --text-cols v2 --label-col v1
```
