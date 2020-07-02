# Practice of Spinning-Up

## Vanilla Policy Gradient

### Training

```
python src/train.py --epochs 3000 --max-ep-len 200
```

Check reward progress:
```
python src/show_rewards.py
```
Note: this code visualizes every results in `results` folder.


### Evaluation

Evaluation with animation:
```
python src/evaluate.py
```


### Reproducing simulation experiments

Training (only for windows):
```
train.bat
```

Rewards visualization:
```
python src/show_results.py
```
