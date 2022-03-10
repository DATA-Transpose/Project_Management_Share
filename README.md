## Hi
## Authorized ssh login

In your computer:
```
$ ssh-keygen ## if you don't have a pair of key.
$ cat ~/.ssh/id_rsa.pub
```

Copy your public key string (the content in `~/.ssh/id_rsa.pub` on your local) into `.ssh/authorized_keys` file on server.

In server:
```
$ mkdir ~/.ssh
$ touch authorized_keys
$ vi authorized_keys
```

Then you can login the server by using `ssh -i <private key dir> <username>@<server address>`
```
$ ssh -i ~/.ssh/id_rsa s4661451@10.35.14.178
```

## Transfering files with scp
`scp -r <from> <to>`

`scp -r <local dir> <username>@<server address>:<remote dir>` where the `<remote dir>` is root at your user home `~`.

```
scp -r ./PeMS-M.zip s4661451@10.35.14.178:./STGCN_R/data
```

Compression
```
$ zstd <file>
$ zstd -d <*.zst file>
```
```
$ tar -I zstd -cvf data.tar.zst ./data/
$ tar -I zstd -xvf data.tar.zst
```

## Jupyter

### Launch the Jupyter Notebook on server and link to your local
After the jupyter server launched you will get the token
```
To access the notebook, open this file in a browser:
    file:///home/s4661451/.local/share/jupyter/runtime/nbserver-349544-open.html
Or copy and paste one of these URLs:
    http://localhost:8888/?token=e3aeb0966b469747d9d881dbb8776a879fb599963d30fe74
    or http://127.0.0.1:8888/?token=e3aeb0966b469747d9d881dbb8776a879fb599963d30fe74
```
Link the remote port to your local, by `ssh -L8888:localhost:8888 <username>@<server address>`
```
$ ssh -L8888:localhost:8888 s4661451@10.35.14.178
```

### If there is no environment name in the kernel list
```
$ conda install -c conda-forge --name comp7812 ipykernel -y
$ python -m ipykernel install --name=comp7812 --user
```


## Run scripts on server
Create `screen` and run in `screen`
```
$ screen -S <name>
$ screen -ls
$ screen -r <name/id>
$ screen -S <id> -X quit
```
Use `ctrl + A + D` to detach current `screen`.

## Tools for monitoring and evaluating computational resources
```
$ free -h ## memory
$ top ## process and cpu
$ nvidia-smi
$ du -sh ## storage occupation for current dir 
$ df -h ## disk
```
