import os
import json
import git

home_dir = os.environ.get("HOME")
os.chdir(f'{home_dir}/Desktop/blech_clust/utils')

model_dir_dict = json.load(open('model_path_params.json','r'))
model_dir = f'{home_dir}/{model_dir_dict["model_path"]}'
repo_url = model_dir_dict['repo_url']

remote_hash = remote_heads = git.cmd.Git().ls_remote(repo_url, heads=True)[:6]

if not os.path.exists(model_dir):
    os.makedirs(model_dir)
    git.Repo.clone_from(repo_url, model_dir)
repo = git.Repo(model_dir) 
origin = repo.remotes.origin
local_hash = repo.head.object.hexsha[:6]
if remote_hash != local_hash: 
    origin.pull()
