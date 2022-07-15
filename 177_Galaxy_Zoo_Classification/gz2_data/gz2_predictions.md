# gz2_predictions catalog : gz2_predictions_0122_v2.csv

| colID |  column  |  type  | description |
|---| -------- | ------ | ----------- |
| 1 | GalaxyID |   int  | Unique galaxy ID of [kaggle GZ2 images](https://www.kaggle.com/jaimetrickz/galaxy-zoo-2-images)| 
| 2 | **class**|  short | 0 round elliptical • 1 in-between elliptical • 2 cigar-shaped elliptical • 3 edge-on spiral • 4 barred spiral • 5 unbarred sprial • 6 irregular • 7 merger |
| 3 | pred_linF|  short | predicted class label from Linformer network (2022.01) |
| 4 | pred_vit |  short | predicted class label from ViT network |
| 5 | pred_res |  short | predicted class label from ResNet50 |
| 6 | vitTresT |  short | 1 if ``pred_vit == class and pred_res == class`` else 0 |
| 7 | vitTresF |  short | 1 if ``pred_vit == class and pred_res != class`` else 0 |
| 8 | vitFresT |  short | 1 if ``pred_vit != class and pred_res == class`` else 0 |
| 9 | vitFresF |  short | 1 if ``pred_vit != class and pred_res != class`` else 0 |
| 10| linTresT |  short | 1 if ``pred_linF == class and pred_res == class`` else 0 |
| 11| linTresF |  short | 1 if ``pred_linF == class and pred_res != class`` else 0 |
| 12| linFresT |  short | 1 if ``pred_linF != class and pred_res == class`` else 0 |
| 13| linFresF |  short | 1 if ``pred_linF != class and pred_res != class`` else 0 |
| 14| dr7objid |  long  | SDSS DR7 object ID |
| 15| dered_u  | double | u band magnitude, corrected for extinction: modelMag-extinction |
| 16| dered_g  | double | g    --  |
| 17| **dered_r** | double | **r band magnitude**, corrected for extinction: modelMag-extinction |
| 18| dered_i  | double | i   -- |
| 19| dered_z  | double | z   -- |
| 20| modelMag_u | double | u band magnitude, better of DeV/Exp magnitude fit |
| 21| modelMag_g | double | g -- |
| 22| modelMag_r | double | r -- |
| 23| modelMag_i | double | i -- |
| 24| modelMag_z | double | z -- |
| 25| lnLDeV_r   | double | DeVaucouleurs fit ln(likelihood) in r-band |
| 26| lnLExp_r   | double | Exponential disk fit ln(likelihood) in r-band |
| 27| petroR50_r | double | Radius containing 50% of Petrosian flux in r-band |
| 28| **petroR90_r** | double | **galaxy size** - Radius containing 90% of Petrosian flux in r-band |
| 29| **dered_g_r** | double | **color** : ``dered_g-dered_r`` |
| 30| model_g_r | double | color : ``modelMag_g-modelMag_r`` | 
| 31| **viewed_edge_on** | double |  GZ2 Task02 -- fraction of votes for edge-on • No : 0 <---> 1 : Yes |
| 32| **anything_odd**   | double |  GZ2 Task06 -- fraction of votes for anything odd? • No : 0 <---> 1 : Yes |
| 33| pred_linF_0921|  short | predicted class label from Linformer network (2021.09) |
| 34| linTresT_0921 |  short | 1 if ``pred_linF_0921 == class and pred_res == class`` else 0 |
| 35| linTresF_0921 |  short | 1 if ``pred_linF_0921 == class and pred_res != class`` else 0 |
| 36| linFresT_0921 |  short | 1 if ``pred_linF_0921 != class and pred_res == class`` else 0 |
| 37| linFresF_0921 |  short | 1 if ``pred_linF_0921 != class and pred_res != class`` else 0 |
------

## Reference Links

- [SDSS DR7 schema browser](http://cas.sdss.org/dr7/en/help/browser/browser.asp?n=ProperMotions&t=U)
