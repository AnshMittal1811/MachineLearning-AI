# [AAAI 2022] FCA: Learning a 3D Full-coverage Vehicle Camouflage for Multi-view Physical Adversarial Attack

## Overview
This is the official implementation and case study of the Full-coverage Vehicle Camouflage(FCA) method proposed in our AAAI 2022 paper [FCA: Learning a 3D Full-coverage Vehicle Camouflage for Multi-view Physical Adversarial Attack](https://arxiv.org/abs/2109.07193). 

Source code can be find in [here](https://github.com/idrl-lab/Full-coverage-camouflage-adversarial-attack/tree/gh-pages/src).

## Abstract
Physical adversarial attacks in object detection have attracted increasing attention. However, most previous works focus on hiding the objects from the detector by generating an individual adversarial patch, which only covers the planar part of the vehicle’s surface and fails to attack the detector in physical scenarios for multi-view, long-distance and partially occluded objects. To bridge the gap between digital attacks and physical attacks, we exploit the full 3D vehicle surface to propose a robust Full-coverage Camouflage Attack (FCA) to fool detectors. Specifically, we first try rendering the nonplanar camouflage texture over the full vehicle surface. To mimic the real-world environment conditions, we then introduce a transformation function to transfer the rendered camouflaged vehicle into a photo realistic scenario. Finally, we design an efficient loss function to optimize the camouflage texture. Experiments show that the full-coverage camouflage attack can not only outperform state-of-the-art methods under various test cases but also generalize to different environments, vehicles, and object detectors. The code of FCA will be available at: https://idrl-lab.github.io/Full-coveragecamouflage adversarial-attack/.

## Framework
![image-20211209204327675](https://gitee.com/freeneuro/PigBed/raw/master/img/image-20211209204327675.png)

## Cases of Digital Attack

### Multi-view Attack: Carmear distance is 3

<table frame=void cellspacing="0" cellpadding="0">
    <tr>
      <td></td>
      <td>Elevation 0</td>
      <td>Elevation 30</td>
      <td>Elevation 50</td>
    </tr>
  <tr>
    <td>Original</td>
    <td><center> <img src = 'https://github.com/winterwindwang/Full-coverage-camouflage-adversarial-attack/blob/gh-pages/assets/distance_3_elevation_0_ori_pred.gif?raw=true'/></center></td>
        <td><center> <img src = 'https://github.com/winterwindwang/Full-coverage-camouflage-adversarial-attack/blob/gh-pages/assets/distance_3_elevation_30_ori_pred.gif?raw=true'/></center></td>
    <td><center> <img src = 'https://github.com/winterwindwang/Full-coverage-camouflage-adversarial-attack/blob/gh-pages/assets/distance_3_elevation_50_ori_pred.gif?raw=true'/></center></td>
  </tr>
    <tr>
    <td>FCA</td>
    <td><center><img src = 'https://github.com/winterwindwang/Full-coverage-camouflage-adversarial-attack/blob/gh-pages/assets/distance_3_elevation_0_adv_pred.gif?raw=true'/></center></td>
    <td><center><img src = 'https://github.com/winterwindwang/Full-coverage-camouflage-adversarial-attack/blob/gh-pages/assets/distance_3_elevation_30_adv_pred.gif?raw=true'/></center></td>    
    <td><center><img src = 'https://github.com/winterwindwang/Full-coverage-camouflage-adversarial-attack/blob/gh-pages/assets/distance_3_elevation_50_adv_pred.gif?raw=true'/></center></td>
  </tr>
</table>


### Multi-view Attack: Carmear distance is 5  

<table border=0>
    <tr>
      <td></td>
      <td>Elevation 20</td>
      <td>Elevation 40</td>
      <td>Elevation 50</td>
    </tr>
   <tr>
      <td>Original</td>
    <td><center> <img src = 'https://github.com/winterwindwang/Full-coverage-camouflage-adversarial-attack/blob/gh-pages/assets/distance_5_elevation_20_ori_pred.gif?raw=true'/></center></td>
     <td><center> <img src = 'https://github.com/winterwindwang/Full-coverage-camouflage-adversarial-attack/blob/gh-pages/assets/distance_5_elevation_40_ori_pred.gif?raw=true'/></center></td>
    <td><center> <img src = 'https://github.com/winterwindwang/Full-coverage-camouflage-adversarial-attack/blob/gh-pages/assets/distance_5_elevation_50_ori_pred.gif?raw=true'/></center></td>
  </tr>
    <tr>
     <td>FCA</td>
    <td><center><img src = 'https://github.com/winterwindwang/Full-coverage-camouflage-adversarial-attack/blob/gh-pages/assets/distance_5_elevation_20_adv_pred.gif?raw=true'/></center></td>
    <td><center><img src = 'https://github.com/winterwindwang/Full-coverage-camouflage-adversarial-attack/blob/gh-pages/assets/distance_5_elevation_40_adv_pred.gif?raw=true'/></center></td> 
    <td><center><img src = 'https://github.com/winterwindwang/Full-coverage-camouflage-adversarial-attack/blob/gh-pages/assets/distance_5_elevation_50_adv_pred.gif?raw=true'/></center></td>
  </tr>
</table>


### Multi-view Attack: Carmear distance is 10

<table>
    <tr>
      <td></td>
      <td>Elevation 30</td>
      <td>Elevation 40</td>
      <td>Elevation 50</td>
    </tr>
    <tr>
      <td>Original</td>
    <td><center> <img src = 'https://github.com/winterwindwang/Full-coverage-camouflage-adversarial-attack/blob/gh-pages/assets/distance_10_elevation_30_ori_pred.gif?raw=true'/></center></td>
    <td><center> <img src = 'https://github.com/idrl-lab//Full-coverage-camouflage-adversarial-attack/blob/gh-pages/assets/distance_10_elevation_40_ori_pred.gif?raw=true'/></center></td>
    <td><center> <img src = 'https://github.com/winterwindwang/Full-coverage-camouflage-adversarial-attack/blob/gh-pages/assets/distance_10_elevation_50_ori_pred.gif?raw=true'/></center></td>
  </tr>
    <tr>
    <td>FCA</td>
    <td><center><img src = 'https://github.com/winterwindwang/Full-coverage-camouflage-adversarial-attack/blob/gh-pages/assets/distance_10_elevation_30_adv_pred.gif?raw=true'/></center></td>
    <td><center><img src = 'https://github.com/idrl-lab/Full-coverage-camouflage-adversarial-attack/blob/gh-pages/assets/distance_10_elevation_40_adv_pred.gif?raw=true'/></center></td>
    <td><center><img src = 'https://github.com/winterwindwang/Full-coverage-camouflage-adversarial-attack/blob/gh-pages/assets/distance_10_elevation_50_adv_pred.gif?raw=true'/></center></td>
  </tr>
</table>

### Multi-view Attack: different distance, elevation and azimuth

<table>
  <tr>
   <td>Original</td>
  <td><center> <img src = 'https://github.com/winterwindwang/Full-coverage-camouflage-adversarial-attack/blob/gh-pages/assets/camera_distance_5_elevation_10_57_ori.png?raw=true' width="100" /></center></td>
    <td><center><img src = 'https://github.com/winterwindwang/Full-coverage-camouflage-adversarial-attack/blob/gh-pages/assets/camera_distance_5_elevation_30_66_ori.png?raw=true' width="100" /> </center></td>
      <td><center><img src = 'https://github.com/winterwindwang/Full-coverage-camouflage-adversarial-attack/blob/gh-pages/assets/camera_distance_10_elevation_0_135_ori.png?raw=true' width="100" /> </center></td>
    <td><center> <img src = 'https://github.com/winterwindwang/Full-coverage-camouflage-adversarial-attack/blob/gh-pages/assets/camera_distance_10_elevation_20_177_ori.png?raw=true' width="100" /> </center></td>
      <td><center>  <img src = 'https://github.com/winterwindwang/Full-coverage-camouflage-adversarial-attack/blob/gh-pages/assets/camera_distance_15_elevation_20_330_ori.png?raw=true' width="100" /> </center></td>
    <td><center><img src = 'https://github.com/winterwindwang/Full-coverage-camouflage-adversarial-attack/blob/gh-pages/assets/camera_distance_15_elevation_50_327_ori.png?raw=true' width="100" /></center></td>
  </tr>
  <tr>
    <td>FCA</td>
  <td><center> <img src = 'https://github.com/winterwindwang/Full-coverage-camouflage-adversarial-attack/blob/gh-pages/assets/camera_distance_5_elevation_10_57_adv.png?raw=true' width="100" /></center></td>
    <td><center><img src = 'https://github.com/winterwindwang/Full-coverage-camouflage-adversarial-attack/blob/gh-pages/assets/camera_distance_5_elevation_30_66_adv.png?raw=true' width="100" /> </center></td>
      <td><center><img src = 'https://github.com/winterwindwang/Full-coverage-camouflage-adversarial-attack/blob/gh-pages/assets/camera_distance_10_elevation_0_135_adv.png?raw=true' width="100"/> </center></td>
    <td><center> <img src = 'https://github.com/winterwindwang/Full-coverage-camouflage-adversarial-attack/blob/gh-pages/assets/camera_distance_10_elevation_20_177_adv.png?raw=true' width="100" /> </center></td>
      <td><center>  <img src = 'https://github.com/winterwindwang/Full-coverage-camouflage-adversarial-attack/blob/gh-pages/assets/camera_distance_15_elevation_20_330_adv.png?raw=true' width="100" /> </center></td>
    <td><center><img src = 'https://github.com/winterwindwang/Full-coverage-camouflage-adversarial-attack/blob/gh-pages/assets/camera_distance_15_elevation_50_327_adv.png?raw=true' width="100"/></center></td>
  </tr>
</table>

### Partial occlusion 

<table>
  <tr>
  <td>Original</td>
  <td><center> <img src = 'https://github.com/winterwindwang/Full-coverage-camouflage-adversarial-attack/blob/gh-pages/assets/camera_distance_1.5_elevation_0_6_ori.png?raw=true'  width="100"/></center></td>
    <td><center><img src = 'https://github.com/winterwindwang/Full-coverage-camouflage-adversarial-attack/blob/gh-pages/assets/camera_distance_1.5_elevation_0_42_ori.png?raw=true' width="100" /> </center></td>
      <td><center><img src = 'https://github.com/winterwindwang/Full-coverage-camouflage-adversarial-attack/blob/gh-pages/assets/camera_distance_1.5_elevation_0_54_ori.png?raw=true' width="100" /> </center></td>
    <td><center> <img src = 'https://github.com/winterwindwang/Full-coverage-camouflage-adversarial-attack/blob/gh-pages/assets/camera_distance_1.5_elevation_0_126_ori.png?raw=true' width="100" /> </center></td>
      <td><center>  <img src = 'https://github.com/winterwindwang/Full-coverage-camouflage-adversarial-attack/blob/gh-pages/assets/camera_distance_1.5_elevation_10_330_ori.png?raw=true' width="100" /> </center></td>
    <td><center><img src = 'https://github.com/winterwindwang/Full-coverage-camouflage-adversarial-attack/blob/gh-pages/assets/camera_distance_1.5_elevation_20_78_ori.png?raw=true' width="100" /></center></td>
  </tr>
  <tr>
    <td>FCA</td>
    <td><center> <img src = 'https://github.com/winterwindwang/Full-coverage-camouflage-adversarial-attack/blob/gh-pages/assets/camera_distance_1.5_elevation_0_6_adv.png?raw=true' width="100" /></center></td>
    <td><center><img src = 'https://github.com/winterwindwang/Full-coverage-camouflage-adversarial-attack/blob/gh-pages/assets/camera_distance_1.5_elevation_0_42_adv.png?raw=true' width="100" /> </center></td>
      <td><center><img src = 'https://github.com/winterwindwang/Full-coverage-camouflage-adversarial-attack/blob/gh-pages/assets/camera_distance_1.5_elevation_0_54_adv.png?raw=true' width="100" /> </center></td>
    <td><center> <img src = 'https://github.com/winterwindwang/Full-coverage-camouflage-adversarial-attack/blob/gh-pages/assets/camera_distance_1.5_elevation_0_126_adv.png?raw=true' width="100" /> </center></td>
      <td><center>  <img src = 'https://github.com/winterwindwang/Full-coverage-camouflage-adversarial-attack/blob/gh-pages/assets/camera_distance_1.5_elevation_10_330_adv.png?raw=true' width="100" /> </center></td>
    <td><center><img src = 'https://github.com/winterwindwang/Full-coverage-camouflage-adversarial-attack/blob/gh-pages/assets/camera_distance_1.5_elevation_20_78_adv.png?raw=true' width="100" /></center></td>
  </tr>
</table>


### Ablation study

#### Different combination of loss terms

<img src = 'https://github.com/winterwindwang/Full-coverage-camouflage-adversarial-attack/blob/gh-pages/assets/abaltion_study_loss.png?raw=true'/>

As we can see from the Figure, different loss terms plays different roles in attacking. For example, the camouflaged car generated by `obj+smooth (we omit the smooth loss, and denotes as obj)` can hidden the vehicle successfully, while the camouflaged car generated by `iou` can successfully suppress the detecting bounding box of the car region, and finally the camouflaged car generated by `cls` successfully make the detector to misclassify the car to anther category.

#### Different initialization ways

<table>
  <tr>
  <td>original </td>
  <td>basic initialization</td>
  <td>random initialization</td>
  <td>zero initialization</td>
  </tr>
  <tr>
  <td><img src = 'https://github.com/winterwindwang/Full-coverage-camouflage-adversarial-attack/blob/gh-pages/assets/data13311_ori.png?raw=true' width="200" /></td>
  <td><img src = 'https://github.com/winterwindwang/Full-coverage-camouflage-adversarial-attack/blob/gh-pages/assets/data13311_adv_basic.png?raw=true' width="200"/></td>
  <td><img src = 'https://github.com/winterwindwang/Full-coverage-camouflage-adversarial-attack/blob/gh-pages/assets/data13311_adv_random.png?raw=true' width="200"/></td>
  <td><img src = 'https://github.com/winterwindwang/Full-coverage-camouflage-adversarial-attack/blob/gh-pages/assets/data13311_adv_zero.png?raw=true' width="200"/></td>
    </tr>
</table>

## Cases of Phyical Attack

### Case in the simulator environment

<iframe src="//player.bilibili.com/player.html?aid=937368019&bvid=BV1dT4y1U75b&cid=550826852&page=1" scrolling="no" border="0" frameborder="no" framespacing="0" allowfullscreen="true"> </iframe>

### Case in the real world

<iframe src="//player.bilibili.com/player.html?aid=852356546&bvid=BV1RL4y1T723&cid=550828041&page=1" scrolling="no" border="0" frameborder="no" framespacing="0" allowfullscreen="true"> </iframe>
