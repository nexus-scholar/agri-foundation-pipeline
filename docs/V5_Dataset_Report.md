# Agri-Foundation V5 Dataset Report

**Date:** 2026-01-13

## 1. Pipeline Overview

This dataset was constructed using a Reproducible Data Pipeline (V5) consisting of five stages:

1.  **Ingestion:** Raw extraction of 10 diverse agricultural datasets (PlantVillage, PlantDoc, RoCoLe, etc.).
2.  **Standardization:** Mapping source-specific folder structures to a unified `Crop_Disease` schema.
3.  **Verification:** Removing corrupt images and deduplicating identical files (MD5 hashing).
4.  **Normalization:** Cleaning naming conventions (e.g., removing `_google` suffixes, fixing typos).
5.  **Sematic Merging:** Consolidating duplicate classes (e.g., `Corn_(maize)` $\to$ `Corn`) and ambiguous labels.

## 2. Source Datasets Summary

| Dataset | Total Images | Description |
| :--- | :--- | :--- |
| Cassava | 21397 | |
| NewPlantDiseases | 87867 | |
| PlantDoc | 2572 | |
| PlantSeg | 11458 | |
| PlantVillage | 41276 | |
| PlantWild | 18542 | |
| TomatoLeaf | 0 | |
| Wheat | 11603 | |

## 3. Detailed Breakdown by Source

### Cassava (21397 images)

| Class | Count |
| --- | --- |
| cassava_bacterial_blight | 1087 |
| cassava_brown_streak_disease | 2189 |
| cassava_green_mottle | 2386 |
| cassava_healthy | 2577 |
| cassava_mosaic_disease | 13158 |


### NewPlantDiseases (87867 images)

| Class | Count |
| --- | --- |
| apple_black_rot | 2484 |
| apple_cedar_rust | 2200 |
| apple_healthy | 2510 |
| apple_scab | 2520 |
| blueberry_healthy | 2270 |
| cherry_(including_sour)_healthy | 2282 |
| cherry_(including_sour)_powdery_mildew | 2104 |
| corn_(maize)_cercospora_spot_gray_spot | 2052 |
| corn_(maize)_common_rust | 2384 |
| corn_(maize)_healthy | 2324 |
| corn_(maize)_northern_blight | 2385 |
| grape_black_rot | 2360 |
| grape_blight_(isariopsis_spot) | 2152 |
| grape_esca_(black_measles) | 2400 |
| grape_healthy | 2115 |
| orange_haunglongbing_(citrus_greening) | 2513 |
| peach_bacterial_spot | 2297 |
| peach_healthy | 2160 |
| pepper_,_bell_bacterial_spot | 2391 |
| pepper_,_bell_healthy | 2485 |
| potato_early_blight | 2424 |
| potato_healthy | 2280 |
| potato_late_blight | 2424 |
| raspberry_healthy | 2226 |
| soybean_healthy | 2527 |
| squash_powdery_mildew | 2170 |
| strawberry_healthy | 2280 |
| strawberry_scorch | 2218 |
| tomato_bacterial_spot | 2127 |
| tomato_early_blight | 2400 |
| tomato_healthy | 2407 |
| tomato_late_blight | 2314 |
| tomato_mold | 2352 |
| tomato_mosaic_virus | 2238 |
| tomato_septoria_spot | 2181 |
| tomato_spider_mites_two-spotted_spider_mite | 2176 |
| tomato_target_spot | 2284 |
| tomato_yellow_curl_virus | 2451 |


### PlantDoc (2572 images)

| Class | Count |
| --- | --- |
| apple_healthy | 91 |
| apple_rust | 88 |
| apple_scab | 93 |
| blueberry_healthy | 115 |
| cherry_healthy | 57 |
| corn_blight | 191 |
| corn_gray_spot | 68 |
| corn_rust | 116 |
| grape_black_rot | 64 |
| grape_healthy | 69 |
| peach_healthy | 111 |
| pepper_bell | 61 |
| pepper_bell_spot | 71 |
| potato_early_blight | 116 |
| potato_late_blight | 105 |
| raspberry_healthy | 119 |
| soybean_healthy | 65 |
| squash_powdery_mildew | 130 |
| strawberry_healthy | 96 |
| tomato_bacterial_spot | 110 |
| tomato_early_blight | 88 |
| tomato_healthy | 63 |
| tomato_late_blight | 111 |
| tomato_mold | 91 |
| tomato_mosaic_virus | 54 |
| tomato_septoria_spot | 151 |
| tomato_two_spotted_spider_mites | 2 |
| tomato_yellow_virus | 76 |


### PlantSeg (11458 images)

| Class | Count |
| --- | --- |
| apple_black_rot | 83 |
| apple_mosaic_virus | 89 |
| apple_rust | 139 |
| apple_scab | 258 |
| banana_banana_anthracnose_baidu | 19 |
| banana_banana_anthracnose_bing | 38 |
| banana_banana_anthracnose_google | 14 |
| banana_banana_black_streak_baidu | 42 |
| banana_banana_black_streak_banana_black_sigatoka_() | 80 |
| banana_banana_black_streak_bing | 44 |
| banana_banana_black_streak_google | 1 |
| banana_banana_bunchy_top_baidu | 40 |
| banana_banana_bunchy_top_bing | 102 |
| banana_banana_bunchy_top_google | 15 |
| banana_banana_cigar_end_rot_bing | 28 |
| banana_banana_cigar_end_rot_google | 32 |
| banana_banana_cordana_spot_baidu | 6 |
| banana_banana_cordana_spot_bing | 40 |
| banana_banana_cordana_spot_google | 9 |
| banana_banana_panama_disease | 63 |
| basil_basil_downy_mildew | 63 |
| bean_bean_halo_blight | 56 |
| bean_bean_mosaic_virus | 58 |
| bean_bean_rust | 115 |
| blueberry_anthracnose_bing | 38 |
| blueberry_anthracnose_google | 4 |
| blueberry_botrytis_blight_bing | 27 |
| blueberry_botrytis_blight_google | 9 |
| blueberry_mummy_berry_bing | 40 |
| blueberry_mummy_berry_google | 7 |
| blueberry_rust | 43 |
| blueberry_scorch_bing | 33 |
| blueberry_scorch_google | 10 |
| broccoli_broccoli_alternaria_spot_bing | 51 |
| broccoli_broccoli_alternaria_spot_google | 14 |
| broccoli_broccoli_downy_mildew | 29 |
| broccoli_broccoli_ring_spot_bing | 14 |
| broccoli_broccoli_ring_spot_google | 2 |
| cabbage_cabbage_alternaria_spot | 61 |
| cabbage_cabbage_black_rot_bing | 87 |
| cabbage_cabbage_black_rot_bing_-_copy | 1 |
| cabbage_cabbage_black_rot_google | 43 |
| cabbage_cabbage_downy_mildew_bing | 79 |
| cabbage_cabbage_downy_mildew_google | 5 |
| carrot_carrot_alternaria_blight_bing | 54 |
| carrot_carrot_alternaria_blight_google | 6 |
| carrot_carrot_cavity_spot | 72 |
| carrot_carrot_cercospora_blight_bing | 19 |
| cauliflower_cauliflower_alternaria_spot | 54 |
| cauliflower_cauliflower_bacterial_soft_rot_bing | 31 |
| cauliflower_cauliflower_bacterial_soft_rot_google | 5 |
| celery_celery_anthracnose | 29 |
| celery_celery_early_blight | 36 |
| cherry_powdery_mildew | 33 |
| cherry_spot | 106 |
| citrus_citrus_canker | 390 |
| citrus_citrus_greening_disease | 133 |
| coffee_coffee_berry_blotch_bing | 100 |
| coffee_coffee_berry_blotch_google | 18 |
| coffee_coffee_black_rot_bing | 3 |
| coffee_coffee_black_rot_google | 4 |
| coffee_coffee_brown_eye_spot_bing | 16 |
| coffee_coffee_brown_eye_spot_google | 4 |
| coffee_coffee_rust | 159 |
| corn_gray_spot | 107 |
| corn_northern_blight | 130 |
| corn_rust | 177 |
| corn_smut | 202 |
| cucumber_cucumber_angular_spot | 182 |
| cucumber_cucumber_bacterial_wilt | 108 |
| cucumber_cucumber_powdery_mildew | 188 |
| eggplant_eggplant_cercospora_spot | 60 |
| eggplant_eggplant_phomopsis_fruit_rot_bing | 33 |
| eggplant_eggplant_phomopsis_fruit_rot_google | 13 |
| eggplant_eggplant_phytophthora_blight_bing | 30 |
| eggplant_eggplant_phytophthora_blight_google | 3 |
| garlic_garlic_blight | 92 |
| garlic_garlic_rust | 106 |
| ginger_ginger_sheath_blight | 68 |
| ginger_ginger_spot | 25 |
| grape_black_rot | 122 |
| grape_downy_mildew | 280 |
| grape_spot | 91 |
| grape_vine_vine_roll_disease | 71 |
| lettuce_lettuce_downy_mildew | 84 |
| lettuce_lettuce_mosaic_virus | 39 |
| maple_maple_tar_spot | 114 |
| peach_anthracnose_bing | 9 |
| peach_anthracnose_google | 4 |
| peach_brown_rot_bing | 154 |
| peach_brown_rot_google | 19 |
| peach_curl | 182 |
| peach_rust_bing | 8 |
| peach_scab_bing | 70 |
| peach_scab_google | 8 |
| pepper_bell_bell_bacterial_spot_bing | 64 |
| pepper_bell_bell_bacterial_spot_google | 12 |
| pepper_bell_bell_blossom_end_rot_bing | 86 |
| pepper_bell_bell_blossom_end_rot_google | 26 |
| pepper_bell_bell_frogeye_spot_bing | 26 |
| pepper_bell_bell_frogeye_spot_google | 5 |
| pepper_bell_bell_powdery_mildew_bing | 21 |
| pepper_bell_bell_powdery_mildew_google | 7 |
| plum_plum_bacterial_spot_bing | 12 |
| plum_plum_bacterial_spot_google | 4 |
| plum_plum_brown_rot_bing | 71 |
| plum_plum_brown_rot_google | 10 |
| plum_plum_pocket_disease | 57 |
| plum_plum_pox_virus_bing | 28 |
| plum_plum_pox_virus_google | 5 |
| plum_plum_rust_bing | 31 |
| plum_plum_rust_google | 3 |
| potato_early_blight | 126 |
| potato_late_blight | 117 |
| raspberry_fire_blight_bing | 25 |
| raspberry_fire_blight_google | 9 |
| raspberry_gray_mold_bing | 33 |
| raspberry_gray_mold_google | 7 |
| raspberry_spot_bing | 13 |
| raspberry_spot_google | 5 |
| raspberry_yellow_rust_bing | 30 |
| raspberry_yellow_rust_google | 7 |
| rice_rice_blast | 83 |
| rice_rice_sheath_blight | 76 |
| soybean_bacterial_blight_baidu | 5 |
| soybean_bacterial_blight_bing | 75 |
| soybean_bacterial_blight_google | 11 |
| soybean_brown_spot_baidu | 10 |
| soybean_brown_spot_bing | 53 |
| soybean_brown_spot_google | 8 |
| soybean_downy_mildew_baidu | 16 |
| soybean_downy_mildew_bing | 100 |
| soybean_downy_mildew_google | 37 |
| soybean_frog_eye_spot_baidu | 14 |
| soybean_frog_eye_spot_bing | 165 |
| soybean_frog_eye_spot_google | 59 |
| soybean_mosaic_baidu | 25 |
| soybean_mosaic_bing | 66 |
| soybean_mosaic_google | 26 |
| soybean_rust_baidu | 21 |
| soybean_rust_bing | 90 |
| soybean_rust_google | 7 |
| squash_powdery_mildew | 182 |
| strawberry_anthracnose | 58 |
| strawberry_scorch | 39 |
| tobacco_tobacco_blue_mold_bing | 47 |
| tobacco_tobacco_blue_mold_google | 13 |
| tobacco_tobacco_brown_spot_bing | 56 |
| tobacco_tobacco_brown_spot_google | 10 |
| tobacco_tobacco_frogeye_spot_bing | 17 |
| tobacco_tobacco_frogeye_spot_google | 11 |
| tobacco_tobacco_mosaic_virus | 41 |
| tomato_bacterial_spot | 109 |
| tomato_early_blight | 187 |
| tomato_late_blight | 163 |
| tomato_mold | 156 |
| tomato_mosaic_virus | 63 |
| tomato_septoria_spot | 130 |
| tomato_yellow_curl_virus | 93 |
| wheat_wheat_bacterial_streak_(black_chaff)_baidu | 28 |
| wheat_wheat_bacterial_streak_(black_chaff)_bing | 27 |
| wheat_wheat_bacterial_streak_(black_chaff)_black_chaff_() | 26 |
| wheat_wheat_bacterial_streak_(black_chaff)_google | 19 |
| wheat_wheat_bacterial_streak_(black_chaff)black_chaff_() | 17 |
| wheat_wheat_head_scab_baidu | 127 |
| wheat_wheat_head_scab_bing | 138 |
| wheat_wheat_head_scab_google | 54 |
| wheat_wheat_loose_smut_baidu | 23 |
| wheat_wheat_loose_smut_bing | 147 |
| wheat_wheat_loose_smut_google | 45 |
| wheat_wheat_powdery_mildew_baidu | 111 |
| wheat_wheat_powdery_mildew_bing | 114 |
| wheat_wheat_powdery_mildew_google | 46 |
| wheat_wheat_rust_baidu | 34 |
| wheat_wheat_rust_bing | 52 |
| wheat_wheat_rust_google | 46 |
| wheat_wheat_septoria_blotch_baidu | 17 |
| wheat_wheat_septoria_blotch_bing | 27 |
| wheat_wheat_septoria_blotch_blotch_() | 126 |
| wheat_wheat_septoria_blotch_google | 41 |
| wheat_wheat_septoria_blotchseptoria_() | 21 |
| wheat_wheat_stem_rust_baidu | 12 |
| wheat_wheat_stem_rust_bing | 108 |
| wheat_wheat_stem_rust_google | 28 |
| wheat_wheat_stripe_rust_baidu | 150 |
| wheat_wheat_stripe_rust_bing | 152 |
| wheat_wheat_stripe_rust_google | 56 |
| zucchini_zucchini_bacterial_wilt_bing | 55 |
| zucchini_zucchini_bacterial_wilt_google | 15 |
| zucchini_zucchini_downy_mildew_bing | 34 |
| zucchini_zucchini_downy_mildew_google | 10 |
| zucchini_zucchini_powdery_mildew_bing | 158 |
| zucchini_zucchini_powdery_mildew_google | 55 |
| zucchini_zucchini_yellow_mosaic_virus | 95 |


### PlantVillage (41276 images)

| Class | Count |
| --- | --- |
| pepper_bell_bacterial_spot | 1994 |
| pepper_bell_healthy | 2956 |
| potato_early_blight | 2000 |
| potato_healthy | 304 |
| potato_late_blight | 2000 |
| tomato_bacterial_spot | 4254 |
| tomato_early_blight | 2000 |
| tomato_healthy | 3182 |
| tomato_late_blight | 3818 |
| tomato_mold | 1904 |
| tomato_mosaic_virus | 746 |
| tomato_septoria_spot | 3542 |
| tomato_spider_mites_two_spotted_spider_mite | 3352 |
| tomato_target_spot | 2808 |
| tomato_yellow_curl_virus | 6416 |


### PlantWild (18542 images)

| Class | Count |
| --- | --- |
| apple_black_rot | 173 |
| apple_healthy | 444 |
| apple_mosaic_virus | 181 |
| apple_rust | 308 |
| apple_scab | 292 |
| banana | 243 |
| banana_panama_disease | 216 |
| basil | 589 |
| basil_downy_mildew | 86 |
| bean | 258 |
| bean_halo_blight | 115 |
| bean_mosaic_virus | 125 |
| bean_rust | 165 |
| blueberry_healthy | 281 |
| blueberry_rust | 117 |
| broccoli | 269 |
| broccoli_downy_mildew | 65 |
| cabbage | 464 |
| cabbage_alternaria_spot | 128 |
| carrot_cavity_spot | 74 |
| cauliflower | 195 |
| cauliflower_alternaria_spot | 98 |
| celery | 212 |
| celery_anthracnose | 44 |
| celery_early_blight | 55 |
| cherry_healthy | 286 |
| cherry_powdery_mildew | 114 |
| cherry_spot | 230 |
| citrus_canker | 535 |
| citrus_greening_disease | 237 |
| coffee | 138 |
| coffee_rust | 190 |
| corn_gray_spot | 216 |
| corn_healthy | 156 |
| corn_northern_blight | 224 |
| corn_rust | 237 |
| corn_smut | 293 |
| cucumber | 348 |
| cucumber_angular_spot | 251 |
| cucumber_bacterial_wilt | 143 |
| cucumber_powdery_mildew | 236 |
| eggplant | 240 |
| eggplant_cercospora_spot | 88 |
| garlic | 228 |
| garlic_blight | 147 |
| garlic_rust | 160 |
| ginger | 177 |
| ginger_sheath_blight | 87 |
| ginger_spot | 87 |
| grape_black_rot | 229 |
| grape_downy_mildew | 395 |
| grape_healthy | 201 |
| grape_spot | 106 |
| grape_vine_roll_disease | 138 |
| lettuce | 244 |
| lettuce_downy_mildew | 118 |
| lettuce_mosaic_virus | 100 |
| maple | 316 |
| maple_tar_spot | 139 |
| peach_curl | 235 |
| peach_healthy | 157 |
| pepper_bell | 240 |
| pepper_bell_spot | 117 |
| plum | 325 |
| plum_pocket_disease | 76 |
| potato_early_blight | 227 |
| potato_healthy | 249 |
| potato_late_blight | 240 |
| raspberry_healthy | 180 |
| rice | 252 |
| rice_blast | 148 |
| rice_sheath_blight | 250 |
| soybean_healthy | 242 |
| squash_healthy | 410 |
| squash_powdery_mildew | 281 |
| strawberry_anthracnose | 98 |
| strawberry_healthy | 199 |
| strawberry_scorch | 76 |
| tobacco | 71 |
| tobacco_mosaic_virus | 91 |
| tomato_bacterial_spot | 280 |
| tomato_early_blight | 346 |
| tomato_healthy | 226 |
| tomato_late_blight | 295 |
| tomato_mold | 239 |
| tomato_mosaic_virus | 189 |
| tomato_septoria_spot | 220 |
| tomato_yellow_curl_virus | 171 |
| zucchini_yellow_mosaic_virus | 181 |


### TomatoLeaf (0 images)

| Class | Count |
| --- | --- |
| tomato_bacterial_spot | 8 |
| tomato_black_spot | 255 |
| tomato_early_blight | 18 |
| tomato_healthy | 798 |
| tomato_late_blight | 160 |
| tomato_mold | 112 |
| tomato_target_spot | 8 |


### Wheat (11603 images)

| Class | Count |
| --- | --- |
| wheat_blackpoint | 2303 |
| wheat_fusariumfootrot | 2250 |
| wheat_healthyleaf | 2250 |
| wheat_leafblight | 2400 |
| wheat_wheatblast | 2400 |


## 4. Final Combined Dataset (V5 Release)

**Total Images:** 161581

**Total Classes:** 167

| Class Name | Image Count |
| --- | --- |
| apple_black_rot | 2654 |
| apple_cedar_rust | 2200 |
| apple_frog_eye_leaf_spot | 3181 |
| apple_healthy | 7658 |
| apple_mosaic_virus | 178 |
| apple_powdery_mildew | 1184 |
| apple_rust | 2251 |
| apple_scab | 7723 |
| banana_anthracnose | 55 |
| banana_black_streak | 84 |
| banana_black_streak_banana_black_sigatoka_() | 74 |
| banana_bunchy_top | 128 |
| banana_cigar_end_rot | 57 |
| banana_cordana_spot | 47 |
| banana_healthy | 243 |
| banana_panama_disease | 212 |
| basil_downy_mildew | 86 |
| basil_healthy | 588 |
| bean_halo_blight | 112 |
| bean_healthy | 257 |
| bean_mosaic_virus | 121 |
| bean_rust | 160 |
| blueberry_anthracnose | 40 |
| blueberry_botrytis_blight | 36 |
| blueberry_healthy | 2662 |
| blueberry_mummy_berry | 46 |
| blueberry_rust | 112 |
| blueberry_scorch | 42 |
| broccoli_alternaria_spot | 62 |
| broccoli_downy_mildew | 65 |
| broccoli_healthy | 269 |
| broccoli_ring_spot | 10 |
| cabbage_alternaria_spot | 120 |
| cabbage_black_rot | 123 |
| cabbage_downy_mildew | 84 |
| cabbage_healthy | 457 |
| carrot_alternaria_blight | 60 |
| carrot_cavity_spot | 74 |
| carrot_cercospora_blight | 17 |
| cassava_bacterial_blight | 1087 |
| cassava_brown_streak_disease | 2189 |
| cassava_green_mottle | 2386 |
| cassava_healthy | 2577 |
| cassava_mosaic_disease | 13158 |
| cauliflower_alternaria_spot | 82 |
| cauliflower_bacterial_soft_rot | 33 |
| cauliflower_healthy | 192 |
| celery_anthracnose | 44 |
| celery_early_blight | 52 |
| celery_healthy | 210 |
| cherry_healthy | 2620 |
| cherry_powdery_mildew | 2212 |
| cherry_spot | 225 |
| citrus_canker | 528 |
| citrus_greening_disease | 237 |
| coffee_berry_blotch | 112 |
| coffee_black_rot | 6 |
| coffee_brown_eye_spot | 14 |
| coffee_healthy | 896 |
| coffee_rust | 187 |
| coffee_unhealthy | 735 |
| corn_gray_spot | 2312 |
| corn_healthy | 2480 |
| corn_northern_blight | 2782 |
| corn_rust | 2726 |
| corn_smut | 292 |
| cucumber_angular_spot | 250 |
| cucumber_bacterial_wilt | 143 |
| cucumber_healthy | 345 |
| cucumber_powdery_mildew | 229 |
| eggplant_cercospora_spot | 80 |
| eggplant_healthy | 240 |
| eggplant_phomopsis_fruit_rot | 46 |
| eggplant_phytophthora_blight | 33 |
| garlic_blight | 145 |
| garlic_healthy | 227 |
| garlic_rust | 152 |
| ginger_healthy | 175 |
| ginger_sheath_blight | 85 |
| ginger_spot | 80 |
| grape_black_rot | 2641 |
| grape_downy_mildew | 388 |
| grape_esca_(black_measles) | 2400 |
| grape_healthy | 2384 |
| grape_spot | 2251 |
| grape_vine_vine_roll_disease | 138 |
| lettuce_downy_mildew | 113 |
| lettuce_healthy | 242 |
| lettuce_mosaic_virus | 91 |
| maple_healthy | 316 |
| maple_tar_spot | 138 |
| orange_huanglongbing_(citrus_greening) | 2513 |
| peach_anthracnose | 13 |
| peach_bacterial_spot | 2297 |
| peach_brown_rot | 170 |
| peach_curl | 235 |
| peach_healthy | 2427 |
| peach_rust | 8 |
| peach_scab | 75 |
| pepper_bell_bacterial_spot | 2461 |
| pepper_bell_bell_blossom_end_rot | 109 |
| pepper_bell_bell_frogeye_spot | 29 |
| pepper_bell_bell_powdery_mildew | 27 |
| pepper_bell_healthy | 2781 |
| pepper_bell_spot | 182 |
| plum_bacterial_spot | 16 |
| plum_brown_rot | 76 |
| plum_healthy | 323 |
| plum_pocket_disease | 75 |
| plum_pox_virus | 32 |
| plum_rust | 34 |
| potato_early_blight | 2748 |
| potato_healthy | 2528 |
| potato_late_blight | 2750 |
| raspberry_fire_blight | 31 |
| raspberry_gray_mold | 38 |
| raspberry_healthy | 2522 |
| raspberry_spot | 18 |
| raspberry_yellow_rust | 36 |
| rice_blast | 148 |
| rice_healthy | 251 |
| rice_sheath_blight | 242 |
| soybean_bacterial_blight | 85 |
| soybean_brown_spot | 53 |
| soybean_downy_mildew | 127 |
| soybean_frog_eye_spot | 204 |
| soybean_healthy | 2827 |
| soybean_mosaic | 92 |
| soybean_rust | 111 |
| squash_healthy | 407 |
| squash_powdery_mildew | 2561 |
| strawberry_anthracnose | 98 |
| strawberry_healthy | 2569 |
| strawberry_scorch | 2292 |
| tobacco_blue_mold | 58 |
| tobacco_brown_spot | 61 |
| tobacco_frogeye_spot | 21 |
| tobacco_healthy | 70 |
| tobacco_mosaic_virus | 90 |
| tomato_bacterial_spot | 2498 |
| tomato_black_spot | 108 |
| tomato_early_blight | 2813 |
| tomato_healthy | 3430 |
| tomato_late_blight | 2765 |
| tomato_mold | 2716 |
| tomato_mosaic_virus | 2473 |
| tomato_septoria_spot | 2528 |
| tomato_target_spot | 2287 |
| tomato_two_spotted_spider_mite | 2178 |
| tomato_yellow_curl_virus | 5423 |
| wheat_bacterial_streak_(black_chaff) | 81 |
| wheat_blackpoint | 1303 |
| wheat_blast | 1171 |
| wheat_fusarium_foot_rot | 1248 |
| wheat_head_scab | 237 |
| wheat_healthy | 1250 |
| wheat_leaf_blight | 1336 |
| wheat_loose_smut | 171 |
| wheat_powdery_mildew | 218 |
| wheat_rust | 97 |
| wheat_septoria_blotch | 184 |
| wheat_stem_rust | 129 |
| wheat_stripe_rust | 285 |
| zucchini_bacterial_wilt | 69 |
| zucchini_downy_mildew | 43 |
| zucchini_powdery_mildew | 203 |
| zucchini_yellow_mosaic_virus | 178 |
