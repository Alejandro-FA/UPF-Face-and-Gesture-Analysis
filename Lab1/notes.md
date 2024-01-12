| Lab                   | Week | Weight                   |
| --------------------- | ---- | ------------------------ |
| 1-2                   | 1-2  | 15%                      |
| 3                     | 3-4  | 15%                      |
| 4                     | 5-6  | 15%                      |
| Competitive challenge | 7-9  | 25% (grade > 5 required) |

No need to train any model. We can use an already existing Viola-Jones like algorithm.


https://docs.opencv.org/4.x/db/d28/tutorial_cascade_classifier.html


## TODO

- If a large face is not detected, try to look for an eye to see ensure that we do not have false positives.

- We could also use `alt2` method for detecting large faces, and take the largest face

- Search for profile faces as well.