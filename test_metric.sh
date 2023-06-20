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
    python Loaded_Tester.py  --seed 42 --model-scale 1.0  --total-runs 5

done
done
done
done
done
