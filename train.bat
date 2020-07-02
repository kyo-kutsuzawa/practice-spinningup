rem Default
start python src/vpg.py --out results/result1 --epochs 3000 --max-ep-len 200

rem NN size
start python src/vpg.py --out results/result6 --epochs 3000 --max-ep-len 200 --model model3
start python src/vpg.py --out results/result5 --epochs 3000 --max-ep-len 200 --model model2

rem discount rate
start python src/vpg.py --out results/result14 --epochs 3000 --max-ep-len 200 --gamma 0.999
start python src/vpg.py --out results/result2  --epochs 3000 --max-ep-len 200 --gamma 0.9
start python src/vpg.py --out results/result15 --epochs 3000 --max-ep-len 200 --gamma 0.7

rem lam
start python src/vpg.py --out results/result13 --epochs 3000 --max-ep-len 200 --lam 0.999
start python src/vpg.py --out results/result8  --epochs 3000 --max-ep-len 200 --lam 0.7

rem learning rate
start python src/vpg.py --out results/result12 --epochs 3000 --max-ep-len 200 --vf-lr 0.01
start python src/vpg.py --out results/result10 --epochs 3000 --max-ep-len 200 --pi-lr 0.003
start python src/vpg.py --out results/result9  --epochs 3000 --max-ep-len 200 --pi-lr 0.003  --vf-lr 0.01
start python src/vpg.py --out results/result11 --epochs 3000 --max-ep-len 200 --vf-lr 0.0001
