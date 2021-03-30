# Transformer-Networks

## docker
```bash
docker run -d -p 2243:22 --gpus all \
  -v /home/z50017127/user/codes/Transformer-Networks:/root/Transformer-Networks \
  -v ~/.keras/:/home/z50017127/.keras/ \
  -v ~/.cache/:/root/.cache/ \
  -it biggan /usr/sbin/sshd -D

ssh -o ServerAliveInterval=30 -o ServerAliveCountMax=2 root@localhost -p 2243

```


