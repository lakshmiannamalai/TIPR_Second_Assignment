# TIPR_Second_Assignment

To Train and Test MNIST: main.py --train ../data/CMNIST --test ../data/MNIST/test --dataset MNIST --configuration '[784 20 20 10]'
   
To Train and Test Cat-Dog: main.py --train ../data/Cat-Dog --test ../data/Cat-Dog/test --dataset Cat-Dog --configuration '[40000 20 20 2]'

To Test MNIST (Model MNIST.npy is assumed to be present in Model directory:
        main.py --test ../data/MNIST/test --dataset MNIST
        
To Test Cat-Dog (Model Cat-Dog.npy is assumed to be present in Model directory:
        main.py --test ../data/Cat-Dog/test --dataset Cat-Dog
