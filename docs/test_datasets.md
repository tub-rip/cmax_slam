## Test on Datasets

### Download datasets

- [ECD dataset](https://rpg.ifi.uzh.ch/davis_data.html): It contains four hand-held rotational motion sequences, which is recorded with a DAVIS240C (240 x 180 px).
  - [shapes_rotation.bag](https://rpg.ifi.uzh.ch/datasets/davis/shapes_rotation.bag)
  - [poster_rotation.bag](https://rpg.ifi.uzh.ch/datasets/davis/poster_rotation.bag)
  - [boxes_rotation.bag](https://rpg.ifi.uzh.ch/datasets/davis/boxes_rotation.bag)
  - [dynamic_rotation.bag](https://rpg.ifi.uzh.ch/datasets/davis/dynamic_rotation.bag)

- [ECRot dataset](https://github.com/tub-rip/ECRot): It contains six synthetic sequences (using a DAVIS240C model) and ten real-world sequences, recorded with a DVXplorer (640 x 480 px resolution).

### Run CMax-SLAM

Change the path to the bag file at the bottom of each launch file, and run the corresponding launch file for the sequence that you want to test on:

- ECD dataset: `roslaunch cmax_slam ijrr.launch` . See [ijrr.launch](https://github.com/tub-rip/cmax_slam/blob/main/launch/ijrr.launch)
- ECRot dataset (synthetic): `roslaunch cmax_slam ecrot_synth.launch` . See [ecrot_synth.launch](https://github.com/tub-rip/cmax_slam/blob/main/launch/ecrot_synth.launch)
- ECRot dataset (hand-held): `roslaunch cmax_slam ecrot_handheld.launch` . See [ecrot_handheld.launch](https://github.com/tub-rip/cmax_slam/blob/main/launch/ecrot_handheld.launch)
- ECRot dataset (motorized mount): `roslaunch cmax_slam ecrot_mount.launch` . See [ecrot_mount.launch](https://github.com/tub-rip/cmax_slam/blob/main/launch/ecrot_mount.launch)
