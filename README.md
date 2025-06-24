# cartpole

### Training Environment Installation
```
git clone https://github.com/Xuanxuan123456789/cartpole.git
conda create --name cartpole python=3.8
conda activate cartpole
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
pip install
pip install tensorboard six
```

### Train
```
cd cartpole
python setup.py develop --CMAKE_PREFIX_PATH /home/xxx/raisim_ws/raisimLib/raisim/linux

cd raisimGymTorch/env/envs/cartpole
python runner.py
```

### Visualizing a policy
```
cd ~/raisim_ws/raisimLib/raisimUnity/linux
./raisimUnity.x86_64

python tester.py -w  /your_runner_data_path
like(python tester.py -w  /home/xxxx/raisimGymTorch/data/cartpole/data/full_400.pt)
```


