#evaluation script                                                                                                                                             
for k in 0
do
for m in 0
do
for p in 0.0
do
for n in CIFAR10
do
for j in 0.05
do
    python Cifar_100.py  --epochs 100 --dataset $n --weight-decay 5e-4 --anneal cosine --batch-size 128 --channel-norm 1 --seed 42 --lr 0.1 --model-scale 1.0 --par-grad-mult 10.0 --par-grad-clip 0.01 --total-runs 5 --momentum 0.9 --test-batch-size 128 --norm-type batch

done
done
done
done
done
