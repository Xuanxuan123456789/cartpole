# cartpole

python setup.py develop --CMAKE_PREFIX_PATH /home/xxx/raisim_ws/raisimLib/raisim/linux

cd raisimGymTorch/env/envs/cartpole
python runner.py


python tester.py -w 

cd ~/raisim_ws/raisimLib/raisimUnity/linux
./raisimUnity.x86_64

python tester.py -w  /home/xxxx/raisimGymTorch/data/cartpole/data/full_400.pt
