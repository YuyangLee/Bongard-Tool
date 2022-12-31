env=dataset

for source in pexels unsplash flickr
do
    tmux new-session -d -s $source
    tmux send-keys 'ca ' $env C-m
    tmux send-keys 'python 0_download.py --source ' $source C-m
    tmux detach -s $source
done
