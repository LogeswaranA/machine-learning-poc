## pip3 install -r requirements.txt

## download nltk using python interactive shell

- import nltk
- nltk.download('punkt_tab')
- nltk.download('stopwords')

## Run main.py

```
python3 train_model.py

```

- It prints the accuracy score
- confusion matrix
- classification report


# Build docker image
```
 docker build -t movie-sentiment-model .  

```

# Run docker 
```
 docker run -p 5000:5000 movie-sentiment-model  

```

# Access Prediction via rest api

