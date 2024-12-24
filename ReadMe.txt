The whole project is divided into three part
1. Train
2. Test
3. FastFFMNet.py

Please see the "How to train.txt" in Train folder and "How to test.txt" in Test folder to train the network and testing

Note that:
The current network architecture is slightly different from that presented in the paper, which is no worries. These are
only minor replacement. e.g. use the groupnorm instead of batchnorm, use the RRelu instead of PRelu.


Issue:
1. The Fast version of the proposed network can not our previous trained model.
This issue is mainly caused by the inconsistency of the checkpoint name between the
network defining and trained model saving due to the modification.
One way to address is to train the Fast version network again,
which only need to replace the corresponding python file in the Train project folder.
The second solutions is to rename the checkpoint name in the trained model.
I will fix this issue as soon as possible.
