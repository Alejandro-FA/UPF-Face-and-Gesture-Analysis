# Notes

Avoid overfitting. Can be useful to plot train and test loss evolution.

Submodels: parallel branches (maximum of 2).

Deep Learning recommended specifically for the feature extractor step

Baseline:

-   Viola-Jones Face Detection
-   HOG Features
-   SVM Classifier / Random Forest

Transfer learning will probably not pass the layer-parameter limitation.

# Ideas

-   Deep Learning limits only apply to the feature extraction step.

-   QUIM recommendation: use a flexible face detection (better to have low false negatives). The face recognition step will handle the face of cats as well in principle. If we use the face detection from the first lab, take into consideration that it had low false positives (but relatively high false negative).

-   Cross validation

-   Multiclass classification vs Binary classification (not recommended by Quim).

-   Use detector from Lab 1 vs use an end-to-end system

-   Leave alignment as a last resource (difficult and not very worth it)

-   Data augmentation

-   Explore transformers

-   Think about how to handle grayscale images. Try to keep all 3 rgb channels

-   Light-weight networks

    -   Light CNN
    -   MobilFace
    -   SqueezeNet
    -   ShuffleNet
    -   Xception

-   Create an id downloader. Keep each id in a separate folder

# Preprocessing models

-   [MTCNN face detector](https://github.com/kpzhang93/MTCNN_face_detection_alignment): recommended in VGGFace2 paper.

# Datasets

-   [DigiFace1M](https://github.com/microsoft/DigiFace1M?tab=readme-ov-file): 720K images with 10K identities (72 images per identity). For each identity, 4 different sets of accessories are sampled and 18 images are rendered for each set.
    500K images with 100K identities (5 images per identity). For each identity, only one set of accessories is sampled. Problem: images are synthetic.
-   [CelebA](https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html). 200K celebrity image with 10k identities. Faces can be downloaded in the wild, or aligned and cropped.
-   [MillionCelebs](https://buptzyb.github.io/MillionCelebs): 22.8M images with 719K identities, downloadable with OneDrive. Faces are already cropped.

-   [IMDB-Face](https://github.com/fwang91/IMDb-Face): 1.7M faces, 59K identities

-   [Labelled Faces in the Wild](http://vis-www.cs.umass.edu/lfw/#download)

-   [MS-Celeb-1](https://github.com/EB-Dodo/C-MS-Celeb) 6M images with 95K identities

# Identities

In this section we include the identities of each of the 80 individuals that are present in the training dataset that we have been given

| ID  | Name                                                                                     |
| --- | ---------------------------------------------------------------------------------------- |
| 1   | Channing Tatum                                                                           |
| 2   | Christina Applegate                                                                      |
| 3   | Richard E. Grant                                                                         |
| 4   | S. Epatha Merkerson                                                                      |
| 5   | Farah Fath                                                                               |
| 6   | Jim Beaver                                                                               |
| 7   | Cheryl Hines                                                                             |
| 8   | Michael Vartan                                                                           |
| 9   | Hayden Christensen                                                                       |
| 10  | Laurence Fishburne                                                                       |
| 11  | Kathryn Joosten                                                                          |
| 12  | Patrick Warburton                                                                        |
| 13  | Jamie Lee Curtis                                                                         |
| 14  | Jason Sudeikis                                                                           |
| 15  | Billy Burke                                                                              |
| 16  | Robert Pattinson                                                                         |
| 17  | Melissa Claire Egan                                                                      |
| 18  | Morena Baccarin                                                                          |
| 19  | Jolene Blalock                                                                           |
| 20  | Matthew Lillard                                                                          |
| 21  | Alicia Goranson                                                                          |
| 22  | Jennie Garth                                                                             |
| 23  | Wanda De Jesus                                                                           |
| 24  | Tracey E. Bregman                                                                        |
| 25  | Tracey Gold                                                                              |
| 26  | Brendan Fraser                                                                           |
| 27  | Kellan Lutz                                                                              |
| 28  | John Travolta                                                                            |
| 29  | Pierce Brosnan                                                                           |
| 30  | Jasmine Guy                                                                              |
| 31  | Swoosie Kurtz                                                                            |
| 32  | Diego Luna                                                                               |
| 33  | Danny Glover                                                                             |
| 34  | David Cross                                                                              |
| 35  | Farrah Fawcett                                                                           |
| 36  | Paul Walker                                                                              |
| 37  | Matt Long                                                                                |
| 38  | Andy García                                                                              |
| 39  | Casey Affleck                                                                            |
| 40  | Carla Gallo                                                                              |
| 41  | James Brolin                                                                             |
| 42  | Christian Bale                                                                           |
| 43  | Nadia Bjorlin                                                                            |
| 44  | Valerie Bertinelli                                                                       |
| 45  | Alec Baldwin                                                                             |
| 46  | Tamara Braun                                                                             |
| 47  | Andy Serkis                                                                              |
| 48  | Jackson Rathbone                                                                         |
| 49  | Robert Redford                                                                           |
| 50  | Julie Marie Berman                                                                       |
| 51  | Chris Kattan                                                                             |
| 52  | Benicio del Toro                                                                         |
| 53  | Anthony Hopkins                                                                          |
| 54  | Lea Michele                                                                              |
| 55  | Jean-Claude Van Damme                                                                    |
| 56  | Adrienne Frantz                                                                          |
| 57  | Kim Fields                                                                               |
| 58  | Wendie Malick                                                                            |
| 59  | Lacey Chabert                                                                            |
| 60  | Harry Connick Jr.                                                                        |
| 61  | Cam Gigandet                                                                             |
| 62  | Andrea Anders                                                                            |
| 63  | Chris Noth                                                                               |
| 64  | Cary Elwes                                                                               |
| 65  | Aisha Hinds                                                                              |
| 66  | Chris Rock                                                                               |
| 67  | Neve Campbell                                                                            |
| 68  | Susan Dey                                                                                |
| 69  | Robert Duvall                                                                            |
| 70  | Caroline Dhavernas                                                                       |
| 71  | Marilu Henner                                                                            |
| 72  | Christian Slater                                                                         |
| 73  | Kris Kristofferson                                                                       |
| 74  | Shelley Long                                                                             |
| 75  | Alan Arkin                                                                               |
| 76  | Faith Ford                                                                               |
| 77  | Jason Bateman                                                                            |
| 78  | Edi Gathegi                                                                              |
| 79  | Emile Hirsch                                                                             |
| 80  | Joaquin Phoenix                                                                          |
