# Description: Load MovieLens data and create a dataset for training and testing.
# Code adapted from https://www.tensorflow.org/federated/tutorials/federated_reconstruction_for_matrix_factorization

from logging import config
import random
import hydra
import requests
import zipfile
import io
import shutil
import pandas as pd
import os
import collections
import matplotlib.pyplot as plt
import torch
import numpy as np
from omegaconf import DictConfig
from torch.utils.data import Dataset, Subset, DataLoader, random_split
from typing import Tuple, Optional, List

#path_to_1m = "/Volumes/T7 Touch/TheseProject/FLDATA/MovieLens"

def download_movielens_data():
    config_dir = os.path.abspath("conf")
    hydra.initialize_config_dir(config_dir=config_dir)
    cfg = hydra.compose(config_name="config_file")
    r = requests.get(cfg.data.data_url)
    z = zipfile.ZipFile(io.BytesIO(r.content))
    z.extractall(path=cfg.data.download_dir)

#download_movielens_data()


def load_movielens_data(
    data_directory: str,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
  """Loads pandas DataFrames for ratings, movies, users from data directory."""
  # Load pandas DataFrames from data directory. Assuming data is formatted as
  # specified in http://files.grouplens.org/datasets/movielens/ml-1m-README.txt.
  ratings_df = pd.read_csv(
      os.path.join(data_directory, "ml-1m", "ratings.dat"),
      sep="::",
      names=["UserID", "MovieID", "Rating", "Timestamp"], engine="python")
  movies_df = pd.read_csv(
      os.path.join(data_directory, "ml-1m", "movies.dat"),
      sep="::",
      names=["MovieID", "Title", "Genres"], engine="python", 
      encoding = "ISO-8859-1")

#   # Create dictionaries mapping from old IDs to new (remapped) IDs for both
#   # MovieID and UserID. Use the movies and users present in ratings_df to
#   # determine the mapping, since movies and users without ratings are unneeded.
  movie_mapping = {
      old_movie: new_movie for new_movie, old_movie in enumerate(
          ratings_df.MovieID.astype("category").cat.categories)
  }
  user_mapping = {
      old_user: new_user for new_user, old_user in enumerate(
          ratings_df.UserID.astype("category").cat.categories)
  }

  # Map each DataFrame consistently using the now-fixed mapping.
  ratings_df.MovieID = ratings_df.MovieID.map(movie_mapping)
  ratings_df.UserID = ratings_df.UserID.map(user_mapping)
  movies_df.MovieID = movies_df.MovieID.map(movie_mapping)

  # Remove nulls resulting from some movies being in movies_df but not
  # ratings_df.
  movies_df = movies_df[pd.notnull(movies_df.MovieID)]

  return ratings_df, movies_df


def plot_genre_distribution(movies_df):
    movie_genres_list = movies_df.Genres.tolist()
    # Count the number of times each genre describes a movie.
    genre_count = collections.defaultdict(int)
    for genres in movie_genres_list:
        curr_genres_list = genres.split('|')
        for genre in curr_genres_list:
            genre_count[genre] += 1
    genre_name_list, genre_count_list = zip(*genre_count.items())

    plt.figure(figsize=(5, 5))
    plt.pie(genre_count_list, labels=genre_name_list)
    plt.title('MovieLens Movie Genres')
    plt.show()
    
    
def print_top_genres_for_user(ratings_df, movies_df, user_id):
  """Prints top movie genres for user with ID user_id."""
  user_ratings_df = ratings_df[ratings_df.UserID == user_id]
  movie_ids = user_ratings_df.MovieID

  genre_count = collections.Counter()
  for movie_id in movie_ids:
    genres_string = movies_df[movies_df.MovieID == movie_id].Genres.tolist()[0]
    for genre in genres_string.split('|'):
      genre_count[genre] += 1

  print(f'\nFor user {user_id}:')
  for (genre, freq) in genre_count.most_common(5):
    print(f'{genre} was rated {freq} times')
    
    
class MovieRatingDataset(Dataset):
    def __init__(self, ratings_df):
        self.x = torch.tensor(ratings_df.MovieID.values, dtype=torch.long)
        self.y = torch.tensor(ratings_df.Rating.values, dtype=torch.float32)
    def __getitem__(self, index) -> Tuple[torch.Tensor, torch.Tensor]:
       return self.x[index], self.y[index]
   
    def __len__(self) -> int:
        return len(self.x)
   

data_dir = "/srv/storage/energyfl@storage1.toulouse.grid5000.fr/FLDATA/movielens"
ratings_df, movies_df = load_movielens_data(data_dir)
grouped_user = ratings_df.groupby('UserID')
grouped_movie = movies_df.groupby('MovieID')
nb_users, nb_movies = len(grouped_user), len(grouped_movie)
print(f"Number of users: {nb_users}, Number of movies: {nb_movies}")
nb_hosts = 10
user_per_host = nb_users // nb_hosts
user_idx = np.arange(nb_users)
np.random.shuffle(user_idx)
partition = np.array_split(user_idx, nb_hosts)

class MovieRatingDataset(Dataset):
    def __init__(self, ratings_df):
        self.users = torch.tensor(ratings_df.UserID.values, dtype=torch.long)
        self.movies = torch.tensor(ratings_df.MovieID.values, dtype=torch.long)
        self.x = torch.stack([self.users, self.movies], dim=1)
        self.y = torch.tensor(ratings_df.Rating.values, dtype=torch.float32)
        
    def __getitem__(self, index) -> Tuple[torch.Tensor, torch.Tensor]:
       return self.x[index], self.y[index]
   
    def __len__(self) -> int:
        return len(self.x)

def create_host_datasets(ratings_df:pd.DataFrame, movies_df:pd.DataFrame, nb_hosts:int, batch_size:int, test_frac:float=0.3, val_frac:float=0.2, shuffle:bool=True)->Tuple[List[DataLoader], List[DataLoader], DataLoader]:
    
    grouped_user = ratings_df.groupby('UserID')
    grouped_movie = movies_df.groupby('MovieID')
    nb_users, nb_movies = len(grouped_user), len(grouped_movie)
    print(f"Number of users: {nb_users}, Number of movies: {nb_movies}")
    test_users = np.random.choice(nb_users, int(test_frac*nb_users), replace=False)
    #TestDATA
    test_data = ratings_df[ratings_df.UserID.isin(test_users)]
    test_dataset = MovieRatingDataset(test_data)
    testloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
    #TrainDATA
    train_data = ratings_df[~ratings_df.UserID.isin(test_users)]
    nb_train_users = len(train_data.UserID.unique())
    users_per_host = nb_train_users // nb_hosts
    user_idx = np.arange(nb_users)
    if shuffle:
        np.random.shuffle(user_idx)
    partition = np.array_split(user_idx, nb_hosts)
    trainloaders = []
    valloaders = []
    for i in range(nb_hosts):
        user_ratings = ratings_df[ratings_df.UserID.isin(partition[i])]
        userid, movieid,rating = user_ratings.UserID.values, user_ratings.MovieID.values, user_ratings.Rating.values   
        user_data = MovieRatingDataset(user_ratings)
        len_val = int(val_frac*len(user_data))
        len_train = len(user_data) - len_val
        train,val = random_split(user_data, [len_train, len_val], generator=torch.Generator().manual_seed(42))
        trainloader = DataLoader(train, batch_size=batch_size, shuffle=True)
        valloader = DataLoader(val, batch_size=batch_size, shuffle=True)
        trainloaders.append(trainloader)
        valloaders.append(valloader)
    return trainloaders,valloaders,testloader


from model import MatrixCompletion
mc = MatrixCompletion(num_users= nb_users, num_items=nb_movies, latent_dim=100)

trainloaders,valloaders,testloader = create_host_datasets(ratings_df, movies_df, nb_hosts=10,batch_size=64)
lss = []
for x,y in trainloaders[0]:
    users_id = x[:,0]
    items_id = x[:,1]
    pred = mc(users_id, items_id)
    loss = torch.nn.functional.mse_loss(pred, y) #+ 0.01*torch.norm(mc.user_embeddings.weight) + 0.01*torch.norm(mc.item_embeddings.weight)
    #print("Loss value: {}".format(loss.item()))
    lss.append(loss.item())
    loss.backward()
    with torch.no_grad():
        for name,params in mc.named_parameters():
            if params.requires_grad:
                params -= 0.1*params.grad 
                params.grad.zero_()
plt.plot(lss)


# def main(cfg:DictConfig):
#     ratings_df, movies_df = load_movielens_data(cfg.data.download_dir)
#     user_datasets = create_user_datasets(ratings_df, min_examples_per_user=50, max_clients=4000)
#     train_users, val_users, test_users = split_dataset(user_datasets, 0.8, 0.1)

# ratings_df, movies_df, user_datasets, train_users, val_users, test_users = main(cfg)

# ratings_df, movies_df = load_movielens_data(path_to_1m)
# user_datasets = create_user_datasets(ratings_df, min_examples_per_user=50, max_clients=4000)
# train_users,val_users, test_users = split_dataset(user_datasets, 0.8, 0.1)

# trainloader, valloader, testloader = create_user_dataloader(user_datasets[1], 0.8, 0.1, 5)
