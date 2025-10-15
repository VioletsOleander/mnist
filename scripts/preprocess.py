# Note: Since the data processing for the original parquet file is too complicate
#  (It involves parsing the possibly compressed binary data inside the parquet file,
# and I spent some time trying to figure it, but it turns out to be more compilcated than I
#  originally thought) Therefore, I decided to
#  use the `datasets` library from Hugging Face to download and preprocess the
#  MNIST dataset, and then save the processed data as numpy files. This script
#  is used to perform the preprocessing and save the numpy files.
