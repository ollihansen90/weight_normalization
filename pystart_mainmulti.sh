docker run --gpus all --name dl -it --rm -v $(pwd):/workingdir -v $HOME:/data --user $(id -u):$(id -g) dl_workingdir_hohansen python3 main_multi.py 2> error.log