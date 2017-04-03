# pyku

Generate a haiku as list containing three lines.


## Installation

```
> virtualenv pyku.env
> source pyku.env
> pip install -r requirements.txt
# download nltk data if you haven't already; this could take a while
> python -m nltk.downloader all
# set location of NLTK_DATA; mine is in my home dir
> export NLTK_DATA=$HOME/nltk_data
```

## Usage

```python
import pyku

haiku = pyku.haiku()
print "\n".join(haiku)
```
