import os
import subprocess
import json
#import git

home_dir = os.environ.get("HOME")
os.chdir(f'{home_dir}/Desktop/blech_clust/utils')

model_dir_dict = json.load(open('model_path_params.json', 'r'))
model_dir = f'{home_dir}/{model_dir_dict["model_path"]}'
download_script_path = os.path.join(
		model_dir, 'src/utils/io/download_models.sh')
process = subprocess.Popen(
	f'bash {download_script_path}', shell=True)
# Forces process to complete before proceeding
stdout, stderr = process.communicate()

#repo_url = model_dir_dict['repo_url']
#
#remote_hash = remote_heads = git.cmd.Git().ls_remote(repo_url, heads=True)[:6]
#
#if not os.path.exists(model_dir):
#    os.makedirs(model_dir)
#    git.Repo.clone_from(repo_url, model_dir)
#repo = git.Repo(model_dir)
#origin = repo.remotes.origin
#local_hash = repo.head.object.hexsha[:6]
#if remote_hash != local_hash:
#    remote_commit_date = origin.repo.head.object.committed_datetime
#    local_commit_date = repo.head.object.committed_datetime
#    print(f'Local commit date: {local_commit_date}')
#    print(f'Remote commit date: {remote_commit_date}')
#    print('Updating model...')
#    origin.pull()
