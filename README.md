# Twitter Sentiment Analysis
This program computes a sentiment analysis over a dataset of more than a million real tweets classified as positive or negative.

## Dataset 
_tweets.zip_ file must be unziped to extract _tweets.csv_, which contains a whole collection of tagged tweets (0: negative, 4: positive).

## Main program
_TSA.py_ preprocesses data by applying:
- Token distinction (phone numbers, HTML tags, usernames, urls, etc.)
- Token normalization (lowercase transformation)
- Punctuation signs normalization (!!! -> exclamations)
- Substitution (@username 123 -> _user_, https://github.com/ -> _url_, etc.)
- Word normalization (perrrrfect -> perfect)
- Negation ("I don't like coffee" -> "I don't NOT_like NOT_coffee")

After that, four different models are trainned and used for classification:
- Naïve Bayes
- Decission tree
- Logistic regression
- Support vector machines
