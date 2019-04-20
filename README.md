# Machine Learning Statistical Utils

## Docker setup for example jupyter notebook

```
docker build -t stat-util .
```

```
docker run --rm -p 8889:8889 -v `pwd`:/workspace stat-util
```
