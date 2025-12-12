# run the application
#nohup python main_domain.py --dataset seq-cifar10 --model LEAR --lr 0.03 --batch_size 32 --n_epochs 10 --num_workers 0 --backbone lear > LEAR.out 2>&1 &
python main_domain.py --dataset seq-cifar10 --model LEAR --lr 0.03 --batch_size 32 --n_epochs 2 --num_workers 0 --backbone lear
